"""
测试 DFormer.py 和 NYUv2_dataset.py API 是否正常工作
验证模型加载、数据集加载、以及推理流程
"""

import torch
import sys
import numpy as np


def test_model_weights_detail():
    """详细检查模型权重是否正确加载"""
    print("=" * 60)
    print(" 模型权重详细检查")
    print("=" * 60)
    
    from get_model.DFormer import get_dformer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_dformer(model_type="base", device=device, freeze=True)
    
    print("\n【权重统计信息】")
    print("-" * 60)
    
    total_params = 0
    zero_params = 0
    nan_params = 0
    inf_params = 0
    
    # 收集各层权重统计
    layer_stats = []
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        # 检查异常值
        zeros = (param == 0).sum().item()
        nans = torch.isnan(param).sum().item()
        infs = torch.isinf(param).sum().item()
        
        zero_params += zeros
        nan_params += nans
        inf_params += infs
        
        # 统计
        mean_val = param.mean().item()
        std_val = param.std().item()
        min_val = param.min().item()
        max_val = param.max().item()
        
        layer_stats.append({
            'name': name,
            'shape': tuple(param.shape),
            'params': num_params,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'zeros': zeros,
            'zero_ratio': zeros / num_params * 100
        })
    
    # 打印总览
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"零值参数: {zero_params:,} ({zero_params/total_params*100:.2f}%)")
    print(f"NaN 参数: {nan_params}")
    print(f"Inf 参数: {inf_params}")
    
    # 权重健康检查
    print("\n【权重健康检查】")
    print("-" * 60)
    
    if nan_params > 0:
        print("⚠️  警告: 存在 NaN 值!")
    else:
        print("✓ 无 NaN 值")
    
    if inf_params > 0:
        print("⚠️  警告: 存在 Inf 值!")
    else:
        print("✓ 无 Inf 值")
    
    # 检查是否像随机初始化（如果mean接近0且std接近某个特定值）
    # 预训练权重通常不会全是接近0的均值
    sample_layers = [s for s in layer_stats if 'weight' in s['name'] and s['params'] > 1000][:10]
    
    all_near_zero_mean = all(abs(s['mean']) < 0.001 for s in sample_layers)
    if all_near_zero_mean and len(sample_layers) > 5:
        print("⚠️  警告: 大部分层均值接近0，可能是随机初始化未加载权重")
    else:
        print("✓ 权重分布看起来正常（非随机初始化）")
    
    # 打印部分层的详细统计
    print("\n【部分层权重详情 (前15层)】")
    print("-" * 60)
    print(f"{'层名称':<50} {'形状':<20} {'均值':>10} {'标准差':>10} {'范围':>20}")
    print("-" * 60)
    
    for stat in layer_stats[:15]:
        name_short = stat['name'][-48:] if len(stat['name']) > 48 else stat['name']
        shape_str = str(stat['shape'])
        range_str = f"[{stat['min']:.3f}, {stat['max']:.3f}]"
        print(f"{name_short:<50} {shape_str:<20} {stat['mean']:>10.4f} {stat['std']:>10.4f} {range_str:>20}")
    
    print(f"\n... 共 {len(layer_stats)} 层")
    
    # 检查特定关键层
    print("\n【关键层检查】")
    print("-" * 60)
    
    key_patterns = ['backbone', 'decoder', 'head', 'cls', 'embed']
    for pattern in key_patterns:
        matching = [s for s in layer_stats if pattern in s['name'].lower()]
        if matching:
            total = sum(s['params'] for s in matching)
            avg_std = np.mean([s['std'] for s in matching])
            print(f"  {pattern}: {len(matching)} 层, {total/1e6:.2f}M 参数, 平均std={avg_std:.4f}")
    
    return True


def test_dataset_detail():
    """详细检查数据集"""
    print("\n" + "=" * 60)
    print(" 数据集详细检查")
    print("=" * 60)
    
    from atk_util.NYUv2_dataset import get_NYUv2_val_loader, get_NYUv2_train_loader, NYUV2_CONFIG
    import os
    
    # 检查数据集路径
    dataset_root = "/root/DFormer/datasets/NYUDepthv2"
    print("\n【数据集路径检查】")
    print("-" * 60)
    
    paths = {
        "RGB": os.path.join(dataset_root, "RGB"),
        "Depth": os.path.join(dataset_root, "Depth"),
        "Label": os.path.join(dataset_root, "Label"),
        "train.txt": os.path.join(dataset_root, "train.txt"),
        "test.txt": os.path.join(dataset_root, "test.txt"),
    }
    
    for name, path in paths.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        if os.path.isdir(path):
            count = len(os.listdir(path))
            print(f"  {status} {name}: {path} ({count} 文件)")
        else:
            print(f"  {status} {name}: {path}")
    
    # 检查文件列表
    print("\n【数据集大小检查】")
    print("-" * 60)
    
    with open(os.path.join(dataset_root, "train.txt"), 'r') as f:
        train_files = [l.strip() for l in f.readlines()]
    with open(os.path.join(dataset_root, "test.txt"), 'r') as f:
        test_files = [l.strip() for l in f.readlines()]
    
    print(f"  训练集: {len(train_files)} 样本 (预期: 795)")
    print(f"  测试集: {len(test_files)} 样本 (预期: 654)")
    
    if len(train_files) == 795:
        print("  ✓ 训练集大小正确")
    else:
        print(f"  ⚠️ 训练集大小不匹配 (期望795, 实际{len(train_files)})")
    
    if len(test_files) == 654:
        print("  ✓ 测试集大小正确")
    else:
        print(f"  ⚠️ 测试集大小不匹配 (期望654, 实际{len(test_files)})")
    
    # 加载并检查数据
    print("\n【数据加载检查】")
    print("-" * 60)
    
    val_loader = get_NYUv2_val_loader(batch_size=1, num_workers=0)
    
    # 检查多个样本
    rgb_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
    depth_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
    label_stats = {'unique': [], 'valid_ratio': []}
    
    num_check = min(10, len(val_loader))
    print(f"  检查前 {num_check} 个样本...")
    
    for i, batch in enumerate(val_loader):
        if i >= num_check:
            break
        
        rgb = batch["data"]
        depth = batch["modal_x"]
        label = batch["label"]
        
        rgb_stats['min'].append(rgb.min().item())
        rgb_stats['max'].append(rgb.max().item())
        rgb_stats['mean'].append(rgb.mean().item())
        rgb_stats['std'].append(rgb.std().item())
        
        depth_stats['min'].append(depth.min().item())
        depth_stats['max'].append(depth.max().item())
        depth_stats['mean'].append(depth.mean().item())
        depth_stats['std'].append(depth.std().item())
        
        valid_mask = label != 255
        label_stats['unique'].append(len(torch.unique(label[valid_mask])))
        label_stats['valid_ratio'].append(valid_mask.float().mean().item())
    
    print("\n【RGB 图像统计】")
    print(f"  形状: [B, 3, 480, 640]")
    print(f"  最小值范围: [{min(rgb_stats['min']):.3f}, {max(rgb_stats['min']):.3f}]")
    print(f"  最大值范围: [{min(rgb_stats['max']):.3f}, {max(rgb_stats['max']):.3f}]")
    print(f"  均值范围: [{min(rgb_stats['mean']):.3f}, {max(rgb_stats['mean']):.3f}]")
    print(f"  标准差范围: [{min(rgb_stats['std']):.3f}, {max(rgb_stats['std']):.3f}]")
    
    # 检查是否归一化
    if min(rgb_stats['min']) < -1 and max(rgb_stats['max']) > 1:
        print("  ✓ RGB 已进行 ImageNet 归一化")
    else:
        print("  ⚠️ RGB 归一化状态异常")
    
    print("\n【Depth 图像统计】")
    print(f"  形状: [B, 3, 480, 640]")
    print(f"  最小值范围: [{min(depth_stats['min']):.3f}, {max(depth_stats['min']):.3f}]")
    print(f"  最大值范围: [{min(depth_stats['max']):.3f}, {max(depth_stats['max']):.3f}]")
    print(f"  均值范围: [{min(depth_stats['mean']):.3f}, {max(depth_stats['mean']):.3f}]")
    print(f"  标准差范围: [{min(depth_stats['std']):.3f}, {max(depth_stats['std']):.3f}]")
    
    if min(depth_stats['min']) < -1 and max(depth_stats['max']) > 1:
        print("  ✓ Depth 已进行归一化")
    else:
        print("  ⚠️ Depth 归一化状态异常")
    
    print("\n【Label 统计】")
    print(f"  形状: [B, 480, 640]")
    print(f"  类别数范围: [{min(label_stats['unique'])}, {max(label_stats['unique'])}] (共40类)")
    print(f"  有效像素比例: [{min(label_stats['valid_ratio']):.2%}, {max(label_stats['valid_ratio']):.2%}]")
    
    if max(label_stats['unique']) <= 40:
        print("  ✓ Label 类别数正确 (<=40)")
    else:
        print("  ⚠️ Label 类别数超出范围")
    
    # 类别名称
    print("\n【类别信息】")
    print(f"  类别数: {NYUV2_CONFIG['num_classes']}")
    print(f"  背景值: {NYUV2_CONFIG['background']}")
    print(f"  前5类: {NYUV2_CONFIG['class_names'][:5]}")
    print(f"  后5类: {NYUV2_CONFIG['class_names'][-5:]}")
    
    return True


def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("1. 测试模块导入")
    print("=" * 60)
    
    try:
        from get_model.DFormer import get_dformer, get_dformerv2, load_from_config
        print("✓ get_model.DFormer 导入成功")
        print(f"  - get_dformer: {get_dformer}")
        print(f"  - get_dformerv2: {get_dformerv2}")
        print(f"  - load_from_config: {load_from_config}")
    except Exception as e:
        print(f"✗ get_model.DFormer 导入失败: {e}")
        return False
    
    try:
        from atk_util.NYUv2_dataset import get_NYUv2_val_loader, get_NYUv2_train_loader, NYUV2_CONFIG
        print("✓ atk_util.NYUv2_dataset 导入成功")
        print(f"  - get_NYUv2_val_loader: {get_NYUv2_val_loader}")
        print(f"  - get_NYUv2_train_loader: {get_NYUv2_train_loader}")
        print(f"  - NYUV2_CONFIG num_classes: {NYUV2_CONFIG['num_classes']}")
    except Exception as e:
        print(f"✗ atk_util.NYUv2_dataset 导入失败: {e}")
        return False
    
    print()
    return True


def test_model_loading():
    """测试模型加载"""
    print("=" * 60)
    print("2. 测试模型加载")
    print("=" * 60)
    
    from get_model.DFormer import get_dformer, get_dformerv2
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 测试 DFormer v1
    try:
        print("\n加载 DFormer v1 (base)...")
        model_v1 = get_dformer(model_type="base", device=device, freeze=True)
        print(f"✓ DFormer v1 加载成功")
        print(f"  - 模型类型: {type(model_v1).__name__}")
        print(f"  - 参数数量: {sum(p.numel() for p in model_v1.parameters()) / 1e6:.2f}M")
        print(f"  - 参数冻结: {not any(p.requires_grad for p in model_v1.parameters())}")
    except Exception as e:
        print(f"✗ DFormer v1 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 DFormer v2 (可选，如果权重存在)
    try:
        print("\n加载 DFormer v2 (base)...")
        model_v2 = get_dformerv2(model_type="base", device=device, freeze=True)
        print(f"✓ DFormer v2 加载成功")
        print(f"  - 模型类型: {type(model_v2).__name__}")
        print(f"  - 参数数量: {sum(p.numel() for p in model_v2.parameters()) / 1e6:.2f}M")
    except FileNotFoundError:
        print("⚠ DFormer v2 权重文件不存在，跳过")
    except Exception as e:
        print(f"⚠ DFormer v2 加载失败 (非致命): {e}")
    
    print()
    return True


def test_dataset_loading():
    """测试数据集加载"""
    print("=" * 60)
    print("3. 测试数据集加载")
    print("=" * 60)
    
    from atk_util.NYUv2_dataset import get_NYUv2_val_loader, NYUV2_CONFIG
    
    try:
        print("\n加载 NYUv2 验证集...")
        val_loader = get_NYUv2_val_loader(batch_size=1, num_workers=0)
        print(f"✓ NYUv2 验证集加载成功")
        print(f"  - 样本数量: {len(val_loader.dataset)}")
        print(f"  - 批次数量: {len(val_loader)}")
        print(f"  - 类别数量: {NYUV2_CONFIG['num_classes']}")
        print(f"  - 图像尺寸: {NYUV2_CONFIG['image_height']}x{NYUV2_CONFIG['image_width']}")
    except Exception as e:
        print(f"✗ NYUv2 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试获取一个batch
    try:
        print("\n获取第一个 batch...")
        batch = next(iter(val_loader))
        
        rgb = batch["data"]
        depth = batch["modal_x"]
        label = batch["label"]
        
        print(f"✓ Batch 获取成功")
        print(f"  - RGB shape: {rgb.shape} dtype: {rgb.dtype}")
        print(f"  - Depth shape: {depth.shape} dtype: {depth.dtype}")
        print(f"  - Label shape: {label.shape} dtype: {label.dtype}")
        print(f"  - RGB 值范围: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"  - Depth 值范围: [{depth.min():.3f}, {depth.max():.3f}]")
        print(f"  - Label 唯一值数量: {len(torch.unique(label))}")
        
    except Exception as e:
        print(f"✗ Batch 获取失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_inference():
    """测试模型推理"""
    print("=" * 60)
    print("4. 测试模型推理")
    print("=" * 60)
    
    from get_model.DFormer import get_dformer
    from atk_util.NYUv2_dataset import get_NYUv2_val_loader
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 加载模型和数据
        print("\n准备模型和数据...")
        model = get_dformer(model_type="base", device=device, freeze=True)
        val_loader = get_NYUv2_val_loader(batch_size=1, num_workers=0)
        
        batch = next(iter(val_loader))
        rgb = batch["data"].to(device)
        depth = batch["modal_x"].to(device)
        label = batch["label"].to(device)
        
        print(f"  - 输入 RGB: {rgb.shape}")
        print(f"  - 输入 Depth: {depth.shape}")
        
        # 前向推理
        print("\n执行前向推理...")
        with torch.no_grad():
            output = model(rgb, depth)
        
        print(f"✓ 推理成功")
        print(f"  - 输出 shape: {output.shape}")
        print(f"  - 输出 dtype: {output.dtype}")
        print(f"  - 输出值范围: [{output.min():.3f}, {output.max():.3f}]")
        
        # 计算预测
        pred = output.argmax(dim=1)
        print(f"  - 预测 shape: {pred.shape}")
        print(f"  - 预测类别数: {len(torch.unique(pred))}")
        
        # 简单准确率
        valid_mask = label != 255
        if valid_mask.sum() > 0:
            acc = (pred[valid_mask] == label[valid_mask]).float().mean()
            print(f"  - 像素准确率: {acc:.4f}")
        
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_atk_workflow():
    """测试 atk.py 中使用的完整工作流"""
    print("=" * 60)
    print("5. 测试 atk.py 工作流兼容性")
    print("=" * 60)
    
    try:
        # 模拟 atk.py 的导入方式
        from get_model.DFormer import get_dformer
        from atk_util.NYUv2_dataset import get_NYUv2_val_loader
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("\n模拟 atk.py 工作流...")
        
        # 1. 加载模型 (与 atk.py 一致)
        model = get_dformer()
        
        # 2. 冻结参数 (与 atk.py 一致)
        for p in model.parameters():
            p.requires_grad = False
        
        # 3. 加载数据 (与 atk.py 一致)
        val_loader = get_NYUv2_val_loader()
        
        # 4. 遍历数据 (与 atk.py 一致)
        for idx, minibatch in enumerate(val_loader):
            images = minibatch["data"].to(device)
            labels = minibatch["label"].to(device)
            modal_xs = minibatch["modal_x"].to(device)
            
            print(f"  - Batch {idx}: images={images.shape}, labels={labels.shape}, modal_xs={modal_xs.shape}")
            
            # 前向推理
            with torch.no_grad():
                logits = model(images, modal_xs)
            
            print(f"  - Logits: {logits.shape}")
            
            # 只测试一个batch
            break
        
        print("\n✓ atk.py 工作流测试通过")
        
    except Exception as e:
        print(f"\n✗ atk.py 工作流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" DFormer API 测试套件")
    print(" 测试 get_model/DFormer.py 和 atk_util/NYUv2_dataset.py")
    print("=" * 60 + "\n")
    
    # 先运行详细检查
    test_model_weights_detail()
    test_dataset_detail()
    
    print("\n" + "=" * 60)
    print(" 功能测试")
    print("=" * 60)
    
    results = {}
    
    # 运行测试
    results["导入测试"] = test_imports()
    
    if results["导入测试"]:
        results["模型加载"] = test_model_loading()
        results["数据集加载"] = test_dataset_loading()
        
        if results["模型加载"] and results["数据集加载"]:
            results["模型推理"] = test_inference()
            results["atk工作流"] = test_atk_workflow()
    
    # 打印总结
    print("=" * 60)
    print(" 测试结果总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有测试通过！API 可以正常工作。")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    # sys.exit(main())

    S:int = 1024
    while True: 
        a : torch.Tensor = torch.randn((S,S),device="cuda")
        b : torch.Tensor = torch.randn((S,S),device="cuda")

        c = (a @ b).mean()

        print(c.item())

