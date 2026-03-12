import csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import DiceSimilarity
from io import StringIO

# 假设CSV文件名为 'drugs.csv'，其中包含两列：药物编码和SMILES序列
csv_filename = 'zhang/drug_list.csv'

# 读取CSV文件并获取SMILES序列
name_list = []
smiles_list = []
with open(csv_filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过标题行
    for row in reader:
        name_list.append(row[0])
        smiles_list.append(row[1])

# 初始化相似度矩阵
num_drugs = len(smiles_list)
similarity_matrix = [[0.0 for _ in range(num_drugs)] for _ in range(num_drugs)]

# 计算每一对SMILES序列的相似度并填充矩阵
for i in range(num_drugs):
    mol1 = Chem.MolFromSmiles(smiles_list[i])
    if mol1 is None:
        continue  # 如果SMILES无效，则跳过
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    for j in range(i + 1, num_drugs):  # 矩阵是对称的，只计算上三角部分
        mol2 = Chem.MolFromSmiles(smiles_list[j])
        if mol2 is None:
            continue  # 如果SMILES无效，则跳过
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        similarity = DiceSimilarity(fp1, fp2)
        similarity_matrix[i][j] = similarity
        similarity_matrix[j][i] = similarity  # 填充下三角部分

# 将相似度矩阵写入CSV文件
output_csv = StringIO()
writer = csv.writer(output_csv)

# 写入标题行
writer.writerow([''] + name_list)

# 写入相似度矩阵数据
for i in range(num_drugs):
    row = [similarity_matrix[i][j] for j in range(num_drugs)]
    writer.writerow([name_list[i]] + row)

# 保存到文件
output_filename = 'zhang/chem_Jacarrd_sim.csv'
with open(output_filename, 'w', newline='') as f:
    f.write(output_csv.getvalue())

print(f'Similarity matrix has been written to {output_filename}')