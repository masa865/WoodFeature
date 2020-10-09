#コピー元のディレクトリから指定した枚数のファイルをランダムに選んで他のディレクトリにコピーするスクリプト
import pathlib
import shutil
import random

#from_dir_pathはコピー元のディレクトリのパス
#to_dir_pathはコピー先のディレクトリのパス
#file_numはコピーするファイルの個数

def copy_file_rand(from_dir_path,to_dir_path,file_num):
    #データセットのルートフォルダのPathオブジェクトを作成
    data_root = pathlib.Path(from_dir_path)

    #ファイルパスのリストを作成
    all_file_paths = list(data_root.glob('*'))
    all_file_paths = [str(path) for path in all_file_paths]

    #シャッフル
    random.shuffle(all_file_paths)

    #file_numの数だけファイルを取り出し新規フォルダにコピー
    need_file_paths = all_file_paths[:file_num]

    toDirPath = pathlib.Path(to_dir_path)
    toDirPath.mkdir(parents=True, exist_ok=True)
    for path in need_file_paths:
        fromFilePath = pathlib.Path(path)
        shutil.copy(fromFilePath,toDirPath)



#---------------test script------------------------------------------
if __name__ == "__main__":
    from_dir_path = r"C:\aha"
    to_dir_path = r"C:\ahe"
    file_num = 3
    copy_file_rand(from_dir_path,to_dir_path,file_num)

