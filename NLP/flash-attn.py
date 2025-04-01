# reference: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py

import torch
import math
import triton
import triton.language as tl


@triton.heuristics(
    {
        "EVEN_M": lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0,
        "EVEN_N": lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0,
        "EVEN_HEADDIM": lambda args: args['headdim'] % args['BLOCK_HEADDIM'] == 0,
    }
)


@trition.jit
def _forward_kernel_fused(
    Q, 
    K, 
    V,
    Bias,
    Out,
    Lse,
    TMP,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    offset_hb = tl.program_id(1) # offset_b * nheads + offset_h
    offset_b = offset_hb // nheads # 第几个批次
    offset_h = offset_hb % nheads # 第几个head
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # query块的偏移数组
    offset_n = tl.arange(0, BLOCK_N) # key块的偏移数组
    offset_d = tl.arange(0, BLOCK_HEADDIM) # head维度偏移数组
    
    # 构造对应的query指针, stride 这些就是用来寻址的
    # 最后有一个广播，最后得到的是 [blcok_m, block_headdim] 的起始指针矩阵
    q_ptrs = Q + offset_b * stride_qb + offset_h * stride_qh + (offset_m[:, None] * stride_qm + offset_d[None, :])
    k_ptrs = K + offset_b * stride_kb + offset_h * stride_kh + (offset_n[:, None] * stride_kn + offset_d[None, :])
    v_ptrs = V + offset_b * stride_vb + offset_h * stride_vh + (offset_n[:, None] * stride_vn + offset_d[None, :])
    
    # 处理偏置项，这些允许更加自由的处理
    # 有两种类型， vector 和 matrix
    if BIAS_TYPE == 'vector':
        # [block_n] boradcast and apply to [block_m, block_n]
        b_ptrs = Bias + offset_b * stride_bb + offset_h * stride_bh + offset_n
    elif BIAS_TYPE == 'matrix':
        # [block_m, block_n] apply to [block_m, block_n]（一个常见的例子是旋转位置编码）
        b_ptrs = Bias + offset_b * stride_bb + offset_h * stride_bh + (offset_m[:, None] * stride_bm + offset_n[None, :])
    
    # 初始化一些常量
    # 临时缓冲区指针，用于存储中间结果（解决编译器 bug）。
    # 查询序列长度（seqlen_q）向上取整后的值，通常是对齐到某个分块大小（如 BLOCK_M 的倍数）。
    # 例如：如果 seqlen_q = 130，BLOCK_M = 64，则 seqlen_q_rounded = 192
    # 批次头组合 + 查询块偏移
    t_ptrs = TMP + offset_hb * seqlen_q_rounded + offset_m # [block_m,]
    # l(x)
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf') # 负无穷初始化
    # m(x)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf') # 初始化
    # 中间变量输出o
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32) # 初始化
    
    # 开始编写恶心的对齐
    # 加载Q
    if EVEN_M & EVEN_N: # 这里必须要这么判断，不能分开判断是因为trition的bug！
        if EVEN_HEADDIM:
            # 所有都是对齐的
            q = tl.load(q_ptrs)
        else:
            # 没有对齐，也就是说当前的headdim < BLOCK_HEADDIM
            # 但是我们q_ptrs的形状是[block_m, block_headdim]
            # 这里加载一个[1,block_headdim]的mask，自动广播
            q = tl.load(q_ptrs, mask=offset_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            # 维度是对齐的，但是 seqlen_q 不一定和 block_m 对齐
            # [block_m, block_headdim]
            # 加载一个[block_m,1]的mask，自动广播
            q = tl.load(q_ptrs, mask=offset_m[:,None] < seqlen_q, other=0.0)
        else:
            # 维度和seqlen_q都不对齐
            q = tl.load(q_ptrs, mask=offset_m[:,None] < seqlen_q & offset_d[None,:] < headdim, other=0.0)
    
    # 开始循环，外层循环是Q，内层循环是K，V
    # 判断当前是否是 causual Attention，如果是则修正能够看到的句子长度
    # 当前能够看到的句子长度的范围是 (start_m + 1) * BLOCK_M
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        # 加载 V
        # 开始写恶心的对齐
        if EVEN_M & EVEN_N:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn) # [block_n, block_headdim] + offset
            
                # [block_n, block_headdim]
                # [1, block_headdim] 进行mask
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offset_d[None,:] < headdim, other=0.0)
        else:
            # seqlen_k 和 block_n 不对齐
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offset_n[:,None] < seqlen_k, other=0.0)
            else:
                # 都不对齐
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offset_n[:,None] < seqlen_k & offset_d[None,:] < headdim, other=0.0)

        # 计算 qk
        # [block_m, block_headdim] * [block_headdim, block_n] -> [block_m, block_n]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        
        # 需要解决对齐的问题
        if not EVEN_N:
            # [block_m, block_n]
            # [1, block_n] 进行一个mask
            qk += tl.where((start_n + offset_n)[None,:] < seqlen_k, 0.0, float('-inf')) # 超出范围的全部设置为负无穷
        # 因果条件
        if IS_CAUSAL:
            # 将未来位置的key mask 掉
            # [block_m, block_n]，使用广播
            # offset_m_{i} >= (start_n + offset_n)_{j}, this is allowed 
            qk += tl.where(offset_m[:,None] >= (start_n + offset_n)[None,:], 0.0, float('-inf'))
        
        # 添加偏置
        if BIAS_TYPE != "none":
            if BIAS_TYPE == 'vector':
                if EVEN_N:
                    # 已经对齐好了
                    # [block_n,]
                    bias = tl.load((b_ptrs + start_n)).to(tl.float32)
                else:
                    # [block_n] 
                    # 构造一个mask， 形状也是 [block_n]
                    bias = tl.load(b_ptrs + start_n, mask=(start_n + offset_n) < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :] # [1, block_n]
            elif BIAS_TYPE == 'matrix':
                # [block_m, block_n]
                if EVEN_N & EVEN_M:
                    bias = tl.load((b_ptrs + start_n)).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, 
                        mask=(offset_m[:,None] < seqlen_q) 
                        & (start_n + offset_n)[None,:] < seqlen_k,
                        other=0.0
                    ).to(tl.float32)
            
            # 一般来说softmax_scale 是 1/sqrt(headdim)
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i) # lse_i是之前的对数和, [block_m]
            p = tl.exp(qk - m_ij[:, None]) # m_ij调整了形状 [block_m, 1]
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        
        l_ij = tl.sum(p, 1) # 在线的新的 l(x)
        acc_o_scale = tl.exp(m_i - m_ij) # 对之前的值的缩放因子
        # 这里存在一个bug，必须要先存进去并且马上进行读取
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None] # 广播
        
        # 加载V
        # 加载逻辑和k一模一样
        if EVEN_M & EVEN_N:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offset_d[None,:] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offset_n)[:,None] < seqlen_k, other=0.0)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn, 
                    mask=(start_n + offset_n)[:,None] 
                    < seqlen_k & offset_d[None,:] < headdim, 
                    other=0.0
                )

        # 这里看论文就知道了
        p = p.to(v.dtype) 
        acc_o += tl.dot(p, v) # 新的输出值
        m_i = m_ij # 更新最大值
        l_i_new = tl.exp(lse_i - m_ij) + l_ij # 目前的新的 sum of e^(-m(x))
        lse_i = m_ij + tl.log(l_i_new) # 转换为了 (x)
    
    # 输出最后还需要scale
    # e^{x}/e^{lse_i} -> e^{x-m_i}*e^{m_i}/e^{lse_i}
    o_scale = tl.exp(m_i - lse_i) # 这里本质上是将safe_softmax -> normal softmax了
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None] # 最后一步才进行总体的scale

    # 重新计算而不是一直存储在寄存器中，释放寄存器空间，防止溢出
    start_m = tl.program_id(0)
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_d = tl.arange(0, BLOCK_HEADDIM)
    lse_ptrs = Lse + offset_hb * seqlen_q_rounded + offset_m
    tl.store(lse_ptrs, lse_i)
    # 把最后的结果写入到HBM中
    out_ptrs = (
        Out
        + offset_b * stride_ob
        + offset_h * stride_oh
        + (offset_m[:, None] * stride_om + offset_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offset_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offset_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offset_m[:, None] < seqlen_q) & (offset_d[None, :] < headdim)
            )

    