class ResultWriter:
    def __init__(self, file_path):
        self.file_path = file_path

    def write_content(self, content):
        with open(self.file_path, 'w') as file:
            file.write(content)
        print("文件写入完成。")
