import pandas as pd


def log_to_excel(log_file, excel_file):
    try:
        # 读取log文件内容
        with open(log_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        print(f'there are {len(lines)} lines in total.')
        
        data = []
        for line in lines:
            try:
                # 将每一行内容解析为字典
                entry = eval(line.strip())
                data.append(entry)
            except SyntaxError:
                print(f"Error parsing line: {line}")

        # 将数据转换为DataFrame
        df = pd.DataFrame(data)

        # 如果epoch列存在，将其移到第一列
        if 'epoch' in df.columns:
            cols = df.columns.tolist()
            cols.insert(0, cols.pop(cols.index('epoch')))
            df = df[cols]

        # 保存为Excel文件
        df.to_excel(excel_file, index=False)
        print(f"Data has been successfully saved to {excel_file}")
    except FileNotFoundError:
        print(f"Error: The file {log_file} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    log_file = '.\\output\\rtdetrv2_organoid\\log.txt'
    excel_file = '.\\output\\rtdetrv2_organoid\\log_data.xlsx'
    log_to_excel(log_file, excel_file)
    