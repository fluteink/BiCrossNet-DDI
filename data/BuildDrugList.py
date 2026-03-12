import pandas as pd
import os

def generate_drug_list(base_path):
    """
    从三个CSV文件中提取药物信息并生成drug_list.csv

    Args:
        base_path: 包含CSV文件的目录路径
    """
    # 定义文件路径
    file_paths = [
        os.path.join(base_path, 'test.csv'),
        os.path.join(base_path, 'train.csv'),
        os.path.join(base_path, 'valid.csv')
    ]

    # 存储所有药物信息的集合
    drug_dict = {}

    # 处理每个文件
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在")
            continue

        print(f"正在处理文件: {file_path}")
        df = pd.read_csv(file_path)

        # 处理每行数据，提取两种药物的信息
        for _, row in df.iterrows():
            # 提取第一种药物
            drugbank_id_1 = row['drugbank_id_1']
            smiles_1 = row['smiles_1']
            if drugbank_id_1 not in drug_dict:
                drug_dict[drugbank_id_1] = smiles_1

            # 提取第二种药物
            drugbank_id_2 = row['drugbank_id_2']
            smiles_2 = row['smiles_2']
            if drugbank_id_2 not in drug_dict:
                drug_dict[drugbank_id_2] = smiles_2

    # 创建新的DataFrame
    drug_data = []
    for drugbank_id, smiles in drug_dict.items():
        drug_data.append({
            'drugbank_id': drugbank_id,
            'smiles': smiles
        })

    drug_df = pd.DataFrame(drug_data)

    # 保存到CSV文件
    output_path = os.path.join(base_path, 'drug_list.csv')
    drug_df.to_csv(output_path, index=False)
    print(f"已生成 {output_path}，共包含 {len(drug_data)} 种药物")

# 使用示例
if __name__ == "__main__":
    base_path = r"E:\Code\pythonProject\reIHM-DDI\data\zhang"
    generate_drug_list(base_path)
