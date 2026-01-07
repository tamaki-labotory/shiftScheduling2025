import tkinter as tk
from tkinter import filedialog
import os

def extract_sections(content):
    """
    テキスト全体から 項目1 と 項目2 の部分のみを抽出する関数
    - 開始: "1. Performance Metrics" を含む行
    - 終了: "3. " で始まる行（ここに来たら終了）
    """
    lines = content.splitlines()
    extracted_lines = []
    is_recording = False
    
    for line in lines:
        # 開始トリガー: 項目1が見つかったら記録開始
        if "1. Performance Metrics" in line:
            is_recording = True
        
        # 終了トリガー: 項目3が見つかったら記録終了（ループも抜ける）
        # strip()で空白を除去して判定を確実にします
        if line.strip().startswith("3.") or "3. Shift Pattern Diversity" in line:
            break
            
        if is_recording:
            extracted_lines.append(line)
            
    # リストを文字列に戻して返す
    return "\n".join(extracted_lines)

def main():
    # --- UI設定 ---
    root = tk.Tk()
    root.withdraw() # メインウィンドウを隠す

    # 1. 入力ファイルの選択
    print("結合するファイル（レポート）を選択してください...")
    input_filepaths = filedialog.askopenfilenames(
        title="結合するレポートファイルを選択（複数選択可）",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )

    if not input_filepaths:
        print("キャンセルされました。")
        return

    # 2. 保存先の指定
    output_filepath = filedialog.asksaveasfilename(
        title="結合後のファイルを保存",
        defaultextension=".txt",
        initialfile="combined_report.txt",
        filetypes=[("Text files", "*.txt")]
    )

    if not output_filepath:
        print("保存先が指定されませんでした。")
        return

    # --- 処理実行 ---
    # 日本語環境のWindowsで作られたファイルなら 'cp932'、VSCode等なら 'utf-8'
    # 読み込みエラーが出る場合はここを変更してください
    encoding_type = 'utf-8' 

    success_count = 0

    try:
        with open(output_filepath, 'w', encoding=encoding_type) as outfile:
            for filepath in input_filepaths:
                filename = os.path.basename(filepath)
                
                # 出力先と同じファイルはスキップ
                if os.path.abspath(filepath) == os.path.abspath(output_filepath):
                    continue

                try:
                    with open(filepath, 'r', encoding=encoding_type) as infile:
                        full_content = infile.read()
                        
                        # ★ここで必要な部分だけ抽出
                        relevant_content = extract_sections(full_content)
                        
                        if relevant_content:
                            # 区切り線とファイル名を書き込む
                            outfile.write("="*60 + "\n")
                            outfile.write(f" FILE: {filename}\n")
                            outfile.write("="*60 + "\n")
                            
                            outfile.write(relevant_content)
                            outfile.write("\n\n") # 次のファイルとの間に空行を入れる
                            
                            print(f"抽出・結合完了: {filename}")
                            success_count += 1
                        else:
                            print(f"スキップ（対象データなし）: {filename}")

                except Exception as e:
                    print(f"読み込みエラー ({filename}): {e}")
        
        print("-" * 30)
        print(f"完了しました。合計 {success_count} ファイルを処理しました。")
        print(f"保存ファイル: {output_filepath}")

    except Exception as e:
        print(f"書き込みエラー: {e}")

if __name__ == "__main__":
    main()