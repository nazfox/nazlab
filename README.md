# Nazuna Laboratory (nazlab)

なづなのためのAI研究フレームワーク


## 特徴(TODO)

- [x] 導入が簡単
- [ ] 学習済みモデルを簡単エクスポート(C++やRustから使えるように)
- [ ] TensorBoardでデータを可視化
- [ ] yamlファイルと学習結果が1対1で対応
- [ ] 学習の中断と再開
- [ ] 第三者にも分かりやすいディレクトリ構造
- [ ] 理解しやすいソースコード
- [ ] 学習結果の再利用


## 使い方

新しいプロジェクトの開始

```
git clone git@github.com:nazfox/nazlab.git project-name
cd project-name
./nazlab init
```

pyenv で Python のインストールと venv で仮想環境を作成を行う

requirements.txt のインストールも行われる


## 利用可能なコマンド一覧

### tensorboard

TensorBoardを実行

ログディレクトリは `./logs`

```
./nazlab tensorboard
```

### notebook

Jupyter Notebook を実行

notebook の保存先は `./notebooks`

```
./nazlab notebook
```

### add

requirements.txt へパッケージを追加  
pip でパッケージをインストール

```
./nazlab add package ...
```


### run

仮想環境内で引数に渡されたコマンドを実行

```
./nazlab run command [args...]
```


### shell

仮想環境内でシェルを起動

```
./nazlab shell
```

### pip-lock

実は requirements.lock を生成出来たりする  

git push するまえに実行したらいいんじゃないかなぁ  
まだ荒削りなのでおすすめはしない

```
./nazlab pip-lock
```

