import csv


def find_max_in_intervals(csv_file_path):
    try:
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            values = []
            for i, row in enumerate(reader):
                # 只处理第1行、第5行、第10行……每四行的数据
                if i % 4 == 0:
                    try:
                        # 将第一列的值转换为整数
                        value = int(row[0])
                        values.append(value)
                    except ValueError:
                        # 忽略无法转换为整数的值
                        continue

            # 找到这些数据的最大值
            max_value = max(values) if values else None

        return max_value
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# 使用示例
csv_file_path = 'assist2017_pid_test.csv'  # 将此路径替换为实际的CSV文件路径
max_value = find_max_in_intervals(csv_file_path)
if max_value is not None:
    print(f"最大值是: {max_value}")
else:
    print("未能找到有效的值。")
