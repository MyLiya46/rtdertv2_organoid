import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_log(log_file):
    # 读取log文件内容
    with open(log_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        try:
            # 将每一行内容解析为字典
            entry = eval(line.strip())
            data.append(entry)
        except SyntaxError:
            print(f"Error parsing line: {line}")

    df = pd.DataFrame(data)

    output_dir = r'D:\Workspace\Organoid_Tracking\organoid_tracking\rtdetrv2_pytorch\test'
    os.makedirs(output_dir, exist_ok=True)

    # 创建一个包含3行1列的子图布局
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # 总训练损失
    key_metrics1 = ['train_loss', 'train_loss_vfl', 'train_loss_bbox', 'train_loss_giou']
    for metric in key_metrics1:
        if metric in df.columns:
            axes[0].plot(df['epoch'], df[metric], label=metric)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Key Training Loss Metrics')
    axes[0].legend()

    # 学习率
    key_metrics2 = ['train_lr']
    for metric in key_metrics2:
        if metric in df.columns:
            axes[1].plot(df['epoch'], df[metric], label=metric)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Learning Rate')
    axes[1].legend()

    # 绘制测试COCO评估指标曲线
    test_columns = ['test_coco_eval_bbox']
    for col in test_columns:
        # 由于test_coco_eval_bbox是列表，这里假设取第一个值进行可视化
        if col in df.columns:
            axes[2].plot(df['epoch'], df[col].str[0], label=col)

    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('COCO Evaluation Metric')
    axes[2].set_title('Test COCO Evaluation Metrics')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'))
    plt.show()

    # 定义不同组的指标
    groups = [
        ['train_loss_vfl_aux_0', 'train_loss_bbox_aux_0', 'train_loss_giou_aux_0'],
        ['train_loss_vfl_aux_1', 'train_loss_bbox_aux_1', 'train_loss_giou_aux_1'],
        ['train_loss_vfl_aux_2', 'train_loss_bbox_aux_2', 'train_loss_giou_aux_2'],
        ['train_loss_vfl_aux_3', 'train_loss_bbox_aux_3', 'train_loss_giou_aux_3'],
        ['train_loss_vfl_aux_4', 'train_loss_bbox_aux_4', 'train_loss_giou_aux_4'],
        ['train_loss_vfl_dn_0', 'train_loss_bbox_dn_0', 'train_loss_giou_dn_0'],
        ['train_loss_vfl_dn_1', 'train_loss_bbox_dn_1', 'train_loss_giou_dn_1'],
        ['train_loss_vfl_dn_2', 'train_loss_bbox_dn_2', 'train_loss_giou_dn_2'],
        ['train_loss_vfl_dn_3', 'train_loss_bbox_dn_3', 'train_loss_giou_dn_3'],
        ['train_loss_vfl_dn_4', 'train_loss_bbox_dn_4', 'train_loss_giou_dn_4'],
        ['train_loss_vfl_dn_5', 'train_loss_bbox_dn_5', 'train_loss_giou_dn_5'],
        ['train_loss_vfl_enc_0', 'train_loss_bbox_enc_0', 'train_loss_giou_enc_0']
    ]

    # 绘制分组指标在子图中
    num_groups = len(groups)
    num_cols = 3
    num_rows = (num_groups + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 5 * num_rows))
    for i, group in enumerate(groups):
        plt.subplot(num_rows, num_cols, i + 1)
        for col in group:
            if col in df.columns:
                if pd.api.types.is_list_like(df[col].iloc[0]):
                    plt.plot(df['epoch'], df[col].str[0], label=col)
                else:
                    plt.plot(df['epoch'], df[col], label=col)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Group {i + 1} Losses')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grouped_loss_metrics.png'))
    plt.show()


if __name__ == "__main__":
    log_file = r'D:\Workspace\Organoid_Tracking\organoid_tracking\rtdetrv2_pytorch\test\log.txt'
    visualize_log(log_file)