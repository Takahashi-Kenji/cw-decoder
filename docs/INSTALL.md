# インストールマニュアル — cw-decoder

AI モールス信号デコーダ **cw-decoder** をご自分のパソコンにインストールして、
起動できるようにするまでの手順です。
パソコンの基本操作ができれば、Python に詳しくなくても順番どおり進めれば完了します。

> このマニュアルは **Windows 11** を前提に書いています。
> コマンドは **PowerShell**（Windows 標準のターミナル）に **1 行ずつコピー＆貼り付け**して
> Enter キーを押してください。

---

## 1. 必要なもの

| 項目 | 内容 |
|---|---|
| パソコン | Windows 11（Windows 10 でも動作します） |
| Python | バージョン **3.11 以上** |
| 受信機 | **任意**。無くてもサンプル音声でデコードを試せます |
| GPU | **不要**。CPU だけで実時間動作します |
| ディスク空き | 約 2 GB（Python パッケージ＋同梱モデル） |
| インターネット | インストール時のパッケージ取得に必要 |

学習済みモデル（`models/full/best_infer.pt`, 約 17 MB）はプログラムに**同梱**されています。
別途ダウンロードや学習は不要で、入手してすぐデコードできます。

---

## 2. Python の準備

### 2-1. インストール済みか確認する

PowerShell を開いて、次のコマンドを実行します。

```powershell
python --version
```

`Python 3.11.x` のように **3.11 以上**が表示されれば準備済みです。次の「3. プログラムの入手」へ進んでください。

`'python' は、内部コマンド…として認識されていません` や、3.10 以下が表示された場合は、次の 2-2 に進みます。

### 2-2. Python をインストールする（未導入の場合）

1. ブラウザで [python.org/downloads](https://www.python.org/downloads/) を開きます。
2. 「Download Python 3.x.x」ボタンからインストーラをダウンロードします。
3. インストーラを起動したら、**最初の画面で必ず `Add python.exe to PATH` にチェック**を入れてから
   `Install Now` を押します。
   （このチェックを忘れると、コマンドで `python` が見つからなくなります）
4. インストール後、**PowerShell を一度閉じて開き直し**、もう一度 `python --version` で確認します。

---

## 3. プログラムの入手

入手方法は 2 通りあります。どちらか片方でかまいません。

### 方法 A：git で取得する（更新が簡単・おすすめ）

git をお使いの場合は、保存したいフォルダで次を実行します。

```powershell
git clone https://github.com/Takahashi-Kenji/cw-decoder.git
cd cw-decoder
```

### 方法 B：ZIP でダウンロードする（git 不要）

1. ブラウザで [github.com/Takahashi-Kenji/cw-decoder](https://github.com/Takahashi-Kenji/cw-decoder) を開きます。
2. 緑色の **`Code`** ボタン → **`Download ZIP`** を選びます。
3. ダウンロードした ZIP を、好きな場所（例：`ドキュメント`）に**展開**します。
4. PowerShell で、展開してできた `cw-decoder` フォルダに移動します。

```powershell
cd $HOME\Documents\cw-decoder
```

> 以降の手順は、**すべて `cw-decoder` フォルダの中**で実行します。
> プロンプトの行頭に `…\cw-decoder>` と表示されていることを確認してください。

---

## 4. 仮想環境の作成

仮想環境とは、このプログラム専用の「箱」を作って、他の Python 環境を汚さないための仕組みです。
作業は最初の 1 回だけです。

### 4-1. 仮想環境を作る

```powershell
python -m venv .venv
```

`cw-decoder` の中に `.venv` フォルダができます。

### 4-2. 仮想環境を有効にする

```powershell
.\.venv\Scripts\Activate.ps1
```

成功すると、プロンプトの行頭に **`(.venv)`** と表示されます。

> **エラーが出たとき**
> `このシステムではスクリプトの実行が無効…` と出たら、次を 1 回実行してから、もう一度 4-2 を試します。
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
>
> 確認を聞かれたら `Y` を入力して Enter を押してください。

---

## 5. インストール

仮想環境が有効（行頭に `(.venv)`）な状態で、次を実行します。
必要なパッケージ（PySide6・PyTorch など）がまとめて入ります。回線によっては数分かかります。

```powershell
pip install -e .
```

最後に `Successfully installed …` と表示されれば完了です。

---

## 6. 動作確認（受信機がなくても試せます）

受信機を接続する前に、**プログラムに付属のサンプル音声**でデコードが正しく動くか確認します。

### 6-1. サンプル音声を生成する

```powershell
python scripts/generate_samples.py --out data/samples --european 12 --japanese 12 --seed 42
```

`data/samples` フォルダに、欧文・和文の合成 CW 音声（WAV）が作られます。

### 6-2. サンプルをデコードする

```powershell
python scripts/decode_demo_samples.py --ckpt models/full/best_infer.pt --dir data/samples
```

最後に次のような行が出れば成功です。**TER（符号の誤り率）が 0.00% 前後**になります。

```
OVERALL  n=24  TER= 0.00%  CER= ...%
```

これで、プログラムとモデルが正しく入っていることが確認できました。

---

## 7. アプリの起動

いよいよデコーダ本体（画面付きアプリ）を起動します。

```powershell
python scripts/run_app.py --ckpt models/full/best_infer.pt
```

「CW デコーダ」というウィンドウが開けば成功です。

実際の操作方法は **[取扱説明書（USAGE.md）](./USAGE.md)** を参照してください。

> **次回以降の起動**
> 2 回目からは、PowerShell で `cw-decoder` フォルダに移動し、
> 仮想環境を有効にしてからアプリを起動します。
>
> ```powershell
> cd $HOME\Documents\cw-decoder
> .\.venv\Scripts\Activate.ps1
> python scripts/run_app.py --ckpt models/full/best_infer.pt
> ```

---

## 8. LLM 清書機能の準備（任意）

LLM 清書機能（AI による誤り訂正・略語展開）を使う場合の設定です。
**使わない場合はこの節をスキップしてください。** アプリ本体のデコードには影響しません。

### 8-1. Ollama を使う（ローカル・無料・API キー不要）

Ollama はご自分のパソコン上で LLM を動かすソフトウェアです。
インターネット経由の API を使わないため、API キーは不要です。

1. ブラウザで [https://ollama.com](https://ollama.com) を開き、**Windows 版をダウンロード**してインストールします。
2. PowerShell でモデルを取得します（初回のみ。サイズは数 GB あります）。

```powershell
ollama pull gemma4:e4b
```

3. Ollama はインストール後は通常自動起動します。起動していないときは次のコマンドで起動します。

```powershell
ollama serve
```

4. アプリ起動後、LLM 清書パネルのプロバイダを `ollama` にすると、**入っているモデルが自動で一覧**されます。モデルを選んで「まとめて清書」を押してください。どのモデルが良いかは [USAGE.md の 8 章](./USAGE.md) を参照。

> API キーの設定（8-2）は**不要**です。

### 8-2. OpenAI または Claude を使う（クラウド・API キー必要）

OpenAI / Claude の API を使う場合は、プロジェクトルートに **`.env`** ファイルを作成して API キーを設定します。

#### 手順 1：`.env.example` を `.env` にコピーする

```powershell
copy .env.example .env
```

（Linux / macOS の場合は `cp .env.example .env`）

#### 手順 2：API キーを取得してファイルに書き込む

メモ帳（または任意のテキストエディタ）で `.env` を開き、取得したキーを記入します。

| キー名 | 取得元 |
|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) のダッシュボード → API keys |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) のダッシュボード → API keys |

**記入例：**

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
```

使わないプロバイダのキーは空欄のまま（`OPENAI_API_KEY=`）でかまいません。

#### 手順 3：アプリでプロバイダを選択する

アプリ起動後、LLM 清書パネルのプロバイダを `openai` または `claude` に設定して「清書」を押してください。

> **`.env` は機密ファイルです。**
> API キーが含まれますので、**他人に渡したり GitHub にアップロードしないでください**。
> このファイルはすでに `.gitignore` に登録されており、通常の `git add` では追跡されません。

---

## 9. 困ったとき（よくある質問）

### Q. `python` が見つからない／認識されない

Python 導入時に **`Add python.exe to PATH` のチェックを忘れた**可能性が高いです。
2-2 の手順でインストーラを再実行し、チェックを入れて入れ直してください。
入れ直し後は PowerShell を開き直します。

### Q. `Activate.ps1 … スクリプトの実行が無効` と出る

Windows の安全機能でスクリプト実行が止められています。4-2 の補足にある
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` を 1 回実行してください。

### Q. `pip install` の途中でエラーになる／止まる

- インターネット接続を確認し、もう一度 `pip install -e .` を実行してください（途中まで入った分はやり直しません）。
- それでも `soxr` や `PySide6` で失敗する場合は、`python --version` が **3.11 以上**か確認してください。
  古いバージョンだと対応するパッケージが見つからないことがあります。

### Q. アプリは起動するが「デコードできない」「変な文字ばかり出る」

`--ckpt models/full/best_infer.pt` を**付け忘れていないか**確認してください。
省略すると未学習モデルで起動し、まともにデコードできません。

### Q. 受信機の音をどう入力するの？

このアプリは**パソコンのマイク／ライン入力**から音を取り込みます。
受信機の音声出力をパソコンの入力端子につなぐか、仮想オーディオ経由で渡してください。
詳しくは取扱説明書の「基本の流れ」を参照してください。

> **重要：リモートデスクトップ（RDP）経由では正しくデコードできません。**
> RDP の音声圧縮で CW の点・線の長さが崩れるためです。
> リアルタイムデコードは、**受信機をつないだパソコンで直接**アプリを動かしてください。

---

## ライセンスについて

本プログラムは MIT ライセンスで公開しています。
画面表示に使う PySide6（Qt）と soxr は LGPL です。通常の利用では問題ありませんが、
将来 PyInstaller 等で単体実行ファイルに固めて再配布する場合のみ LGPL の条件にご注意ください。

---

## 受信機を繋いだ PC が別のとき（LAN 経由）

GPU を積んだ PC と受信機を繋いだ PC が別な場合、音声を LAN 越しに送れます。

**リモートデスクトップのマイク転送は使わないでください。** 音声が狭帯域
コーデックで圧縮され、エネルギーの 99% が 1000 Hz 以下になります。

### 受信機を繋いだ PC で

```
python scripts/audio_send.py --list       # 入力デバイス番号を調べる
python scripts/audio_send.py --device 13  # 送信開始 (Ctrl+C で終了)
```

起動すると、GPU 側で打つコマンドがそのまま表示されます。

### GPU を積んだ PC で

```
python scripts/run_app.py --ckpt models/full/best_infer.pt --net-source 192.168.1.20
```

`--net-source` を指定すると入力デバイス選択は無効になります。ポートを変えた
場合は `--net-source 192.168.1.20:45000` のように書きます。

送信側を止めて起動し直しても、GPU 側は操作なしで復帰します。

> **BPF を必ず ON にしてください。** BPF 未通過の音をモデルに入れると認識が
> 崩壊します（実測で TER 97%）。送信側は生の音を流し、整形は cw-decoder 側で
> 行う設計です。

> 音声は暗号化せずに流れます。家庭内 LAN での利用を前提としており、外部へ
> 公開するポートには割り当てないでください。
