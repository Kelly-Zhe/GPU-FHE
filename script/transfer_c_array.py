import numpy as np
import re
import json


def parse_cpp_arrays_and_variables(file_content):
    # 匹配C++数组的正则表达式，包括 uint64_t 和 double 类型，且不要求是 const
    array_pattern = re.compile(
        r'static\s+(?:const\s+)?(uint64_t|double)\s+(\w+)\s*'  # 匹配类型和数组名
        r'((?:\[\d+\])+)\s*'  # 匹配数组的维度和大小
        r'=\s*\{(.*?)\};', re.DOTALL)  # 匹配等号和花括号中的内容

    # 匹配C++变量的正则表达式
    variable_pattern = re.compile(
        r'static\s+const\s+(?:uint32_t|int)\s+(\w+)\s*=\s*(\d+);'
    )

    arrays = {}
    variables = {}

    # 解析数组
    for match in array_pattern.finditer(file_content):
        dtype = match.group(1)  # 获取数据类型
        name = match.group(2)  # 获取数组名
        dimensions = match.group(3)  # 获取数组的维度信息
        data_str = match.group(4)  # 获取数组数据部分

        # 解析维度信息
        shape = [int(x[1:-1]) for x in re.findall(r'\[\d+\]', dimensions)]
        total_elements = np.prod(shape)  # 计算数组总元素个数

        # 解析数组数据部分，保留嵌套花括号
        data_str = re.sub(r'\s+', '', data_str)  # 移除空格和换行符
        data_str = '[' + data_str + ']'  # 添加最外层的方括号

        # 替换内部花括号为方括号
        data_str = data_str.replace('{', '[').replace('}', ']')

        # 移除多余的逗号
        data_str = re.sub(r',\s*(?=\])', '', data_str)

        try:
            # 转换数据字符串为列表
            data_list = json.loads(data_str)

            # 转换为numpy数组
            if dtype == 'uint64_t':
                data = np.array(data_list, dtype=np.uint64)
            elif dtype == 'double':
                data = np.array(data_list, dtype=np.float64)

            # 确认数据大小匹配
            if data.size != total_elements:
                raise ValueError(f"Data size mismatch for array {name}: expected {total_elements}, got {data.size}")

            # 重塑数据为正确的维度
            array = data.reshape(shape)

            # 保存到字典中
            arrays[name] = array

            # 打印调试信息
            print(f"Parsed array {name}: {array}")
        except Exception as e:
            print(f"Error processing array {name}: {e}")

    # 解析变量
    for match in variable_pattern.finditer(file_content):
        name = match.group(1)  # 获取变量名
        value = int(match.group(2))  # 获取变量值
        variables[name] = np.uint64(value)

        # 打印调试信息
        print(f"Parsed variable {name}: {value}")

    return arrays, variables


def read_cpp_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def write_numpy_arrays_and_variables(arrays, variables, output_file):
    with open(output_file, 'w') as file:
        file.write("import numpy as np\n\n")
        for name, array in arrays.items():
            dtype = 'np.uint64' if array.dtype == np.uint64 else 'np.float64'
            file.write(f'{name} = np.array({array.tolist()}, dtype={dtype})\n')
        for name, value in variables.items():
            file.write(f'{name} = np.uint64({value})\n')


def process_file(input_file, output_file):
    # 读取C++文件内容
    cpp_file_content = read_cpp_file(input_file)
    # 解析C++数组和变量并转换为numpy数组和变量
    numpy_arrays, numpy_variables = parse_cpp_arrays_and_variables(cpp_file_content)
    # 将结果写入Python文件
    write_numpy_arrays_and_variables(numpy_arrays, numpy_variables, output_file)


# 处理所有文件
file_mappings = {
    'bsConst.txt': '../data/bsConst.py',
    'data.txt': '../data/params_N256_L4_P1.py',
    'data_N256_L4_P2.txt': '../data/params_N256_L4_P2.py',
    'data_N64.txt': '../data/params_N64.py',
}

for input_file, output_file in file_mappings.items():
    process_file(input_file, output_file)
