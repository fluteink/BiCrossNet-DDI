import pandas as pd


# 读取相似度矩阵CSV文件
names = ['indication', 'offsideeffect', 'sideeffect', 'transporter']

for name in names:
    similarity_matrix_file = f'zhang/{name}_Jacarrd_sim.csv'
    df = pd.read_csv(similarity_matrix_file, index_col=0)

    # 读取药物顺序CSV文件
    drug_order_file = 'zhang/drug_list.csv'
    drug_order_df = pd.read_csv(drug_order_file)

    # 假设药物名称在药物顺序CSV文件中的'Drug'列
    drug_order = drug_order_df['cid'].tolist()

    # 确保药物顺序中的药物在相似度矩阵中都存在
    drug_order = [drug for drug in drug_order if drug in df.index]

    # 根据药物顺序重新排列相似度矩阵
    new_df = df.loc[drug_order, drug_order]

    # 将重新排列后的相似度矩阵保存到新的CSV文件
    new_similarity_matrix_file = f'{name}_Jacarrd_sim.csv'
    new_df.to_csv(new_similarity_matrix_file)

    print(f"重新排列后的相似度矩阵已保存到 {new_similarity_matrix_file}")
