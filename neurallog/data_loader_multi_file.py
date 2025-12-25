"""
多文件日志数据加载模块

用于处理包含多个独立日志文件的数据集。
每个日志文件独立处理，使用滑动窗口构建序列，然后合并所有序列。
"""
import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from sklearn.utils import shuffle

from neurallog.data_loader import (
    clean, bert_encoder, xlm_encoder, gpt2_encoder, balancing
)


def find_log_files(data_root: str, pattern: str = "*log*.txt") -> List[str]:
    """
    递归查找所有匹配的日志文件
    
    参数:
        data_root: str, 数据根目录路径
        pattern: str, 文件匹配模式（默认 "*log*.txt"）
    
    返回:
        List[str], 找到的日志文件路径列表（按字母顺序排序）
    """
    data_path = Path(data_root)
    if not data_path.exists():
        raise ValueError(f"数据根目录不存在: {data_root}")
    
    # 递归查找所有匹配的文件
    log_files = list(data_path.rglob(pattern))
    log_files = [str(f) for f in log_files]
    log_files.sort()  # 按路径排序，保证可重复性
    
    print(f"找到 {len(log_files)} 个日志文件")
    return log_files


def load_single_file_sequences(
    log_file: str,
    windows_size: int = 20,
    step_size: int = 0,
    e_type: str = 'bert',
    no_word_piece: int = 0,
    encoder_cache: Optional[dict] = None
) -> Tuple[List, List]:
    """
    从单个日志文件加载序列
    
    参数:
        log_file: str, 日志文件路径
        windows_size: int, 滑动窗口大小
        step_size: int, 滑动窗口步长（0表示固定窗口）
        e_type: str, 嵌入类型（'bert', 'xlm', 'gpt2'）
        no_word_piece: int, 是否使用WordPiece分词
        encoder_cache: dict, 编码器缓存（跨文件共享，避免重复编码相同日志）
    
    返回:
        tuple: (sequences, labels)
            sequences: List, 日志序列列表，每个序列是一个包含多个768维向量的列表
            labels: List, 标签列表，0=正常，1=异常
    """
    # 选择编码器
    e_type = e_type.lower()
    if e_type == "bert":
        encoder = bert_encoder
    elif e_type == "xlm":
        encoder = xlm_encoder
    elif e_type == "gpt2":
        encoder = gpt2_encoder
    else:
        raise ValueError(f'嵌入类型 {e_type.upper()} 不在 BERT, XLM, GPT2 中')
    
    # 初始化编码器缓存（如果未提供）
    if encoder_cache is None:
        encoder_cache = {}
    
    # 读取日志文件
    with open(log_file, mode="r", encoding='utf8') as f:
        logs = f.readlines()
        logs = [x.strip() for x in logs if x.strip()]  # 移除空行
    
    if len(logs) < windows_size:
        print(f"警告: 文件 {log_file} 的日志行数 ({len(logs)}) 少于窗口大小 ({windows_size})，跳过")
        return [], []
    
    sequences = []
    labels = []
    
    # 使用滑动窗口构建序列
    i = 0
    while i <= len(logs) - windows_size:
        seq = []
        label = 0
        
        # 构建一个窗口的序列
        for j in range(i, i + windows_size):
            # 判断是否为异常（假设格式与BGL类似：'-'开头表示正常）
            # 如果您的日志格式不同，需要修改这里的判断逻辑
            if logs[j] and logs[j][0] != "-":
                label = 1
            
            # 提取日志内容（移除标签字符）
            content = logs[j]
            if content.find(' ') != -1:
                content = content[content.find(' ') + 1:]  # 移除第一个空格前的部分
            
            # 预处理日志
            content = clean(content.lower())
            
            # 使用缓存避免重复编码
            if content not in encoder_cache:
                try:
                    encoder_cache[content] = encoder(content, no_word_piece)
                except Exception as e:
                    print(f"编码失败: {content[:50]}... 错误: {e}")
                    encoder_cache[content] = np.zeros((768,))
            
            emb = encoder_cache[content]
            seq.append(emb)
        
        sequences.append(seq)
        labels.append(label)
        
        # 移动窗口
        if step_size == 0:
            step_size = windows_size  # 固定窗口
        i += step_size
    
    return sequences, labels


def load_multi_file_logs(
    data_root: str,
    pattern: str = "*log*.txt",
    train_ratio: float = 0.8,
    windows_size: int = 20,
    step_size: int = 0,
    e_type: str = 'bert',
    mode: str = "balance",
    no_word_piece: int = 0,
    file_train_ratio: Optional[float] = None
) -> Tuple[Tuple, Tuple]:
    """
    从多个独立日志文件加载数据
    
    该函数递归遍历数据根目录，找到所有匹配的日志文件，对每个文件独立处理，
    使用滑动窗口构建序列，然后合并所有序列。
    
    关键点：
    - 每个文件独立处理，保持文件内的日志顺序
    - 位置编码仍然有意义，因为它是相对于序列内的位置（0到max_len-1）
    - 不同文件的序列可以混合训练，位置编码互不干扰
    
    参数:
        data_root: str, 数据根目录路径
        pattern: str, 文件匹配模式（默认 "*log*.txt"）
        train_ratio: float, 训练集比例（例如0.8）
        windows_size: int, 滑动窗口大小
        step_size: int, 滑动窗口步长（0表示固定窗口，等于windows_size）
        e_type: str, 嵌入类型（'bert', 'xlm', 'gpt2'）
        mode: str, 数据平衡模式（'balance' 或 'imbalance'）
        no_word_piece: int, 是否使用WordPiece分词
        file_train_ratio: Optional[float], 文件级别的训练集比例
            如果为None，则按序列级别分割
            如果指定（如0.8），则80%的文件用于训练，20%用于测试
    
    返回:
        tuple: ((x_tr, y_tr), (x_te, y_te))
            x_tr: 训练集特征，列表，每个元素是一个日志序列
            y_tr: 训练集标签，列表，0=正常，1=异常
            x_te: 测试集特征
            y_te: 测试集标签
    """
    print("=" * 60)
    print("多文件日志数据加载")
    print("=" * 60)
    
    # 查找所有日志文件
    log_files = find_log_files(data_root, pattern)
    
    if len(log_files) == 0:
        raise ValueError(f"在 {data_root} 中未找到匹配 {pattern} 的文件")
    
    # 共享的编码器缓存（跨文件共享，提高效率）
    encoder_cache = {}
    
    all_sequences = []
    all_labels = []
    file_info = []  # 记录每个序列来自哪个文件（可选，用于调试）
    
    # 处理每个文件
    for file_idx, log_file in enumerate(log_files):
        print(f"\n处理文件 [{file_idx+1}/{len(log_files)}]: {log_file}")
        
        sequences, labels = load_single_file_sequences(
            log_file=log_file,
            windows_size=windows_size,
            step_size=step_size,
            e_type=e_type,
            no_word_piece=no_word_piece,
            encoder_cache=encoder_cache
        )
        
        if len(sequences) > 0:
            all_sequences.extend(sequences)
            all_labels.extend(labels)
            file_info.extend([file_idx] * len(sequences))
            print(f"  生成 {len(sequences)} 个序列，其中 {sum(labels)} 个异常")
        else:
            print(f"  跳过（日志行数不足）")
    
    print(f"\n总共生成 {len(all_sequences)} 个序列")
    print(f"唯一日志消息数: {len(encoder_cache)}")
    print(f"异常序列数: {sum(all_labels)}")
    print(f"正常序列数: {len(all_labels) - sum(all_labels)}")
    
    # 分割训练集和测试集
    if file_train_ratio is not None:
        # 按文件级别分割
        print(f"\n按文件级别分割（文件训练比例: {file_train_ratio}）")
        n_train_files = int(len(log_files) * file_train_ratio)
        train_file_indices = set(range(n_train_files))
        
        x_tr, y_tr = [], []
        x_te, y_te = [], []
        
        for i, (seq, label, file_idx) in enumerate(zip(all_sequences, all_labels, file_info)):
            if file_idx in train_file_indices:
                x_tr.append(seq)
                y_tr.append(label)
            else:
                x_te.append(seq)
                y_te.append(label)
    else:
        # 按序列级别分割（默认）
        print(f"\n按序列级别分割（序列训练比例: {train_ratio}）")
        all_sequences, all_labels = shuffle(all_sequences, all_labels)
        
        n_train = int(len(all_sequences) * train_ratio)
        x_tr = all_sequences[:n_train]
        y_tr = all_labels[:n_train]
        x_te = all_sequences[n_train:]
        y_te = all_labels[n_train:]
    
    # 数据平衡（如果启用）
    if mode == 'balance':
        print("\n对训练集进行类别平衡...")
        x_tr, y_tr = balancing(x_tr, y_tr)
    
    # 打印统计信息
    num_train = len(x_tr)
    num_test = len(x_te)
    num_total = num_train + num_test
    num_train_pos = sum(y_tr)
    num_test_pos = sum(y_te)
    num_pos = num_train_pos + num_test_pos
    
    print("\n" + "=" * 60)
    print("数据加载完成")
    print("=" * 60)
    print(f'总计: {num_total} 个序列, {num_pos} 个异常, {num_total - num_pos} 个正常')
    print(f'训练集: {num_train} 个序列, {num_train_pos} 个异常, {num_train - num_train_pos} 个正常')
    print(f'测试集: {num_test} 个序列, {num_test_pos} 个异常, {num_test - num_test_pos} 个正常')
    print("=" * 60)
    
    return (x_tr, y_tr), (x_te, y_te)

