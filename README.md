# Nazuna Laboratory (nazlab)

## 要件

- 学習済みモデルを簡単エクスポート
- TensorBoardで可視化
- yamlファイルと学習結果が1対1で対応
- 学習の中断と再開が可能
- モジュール化を意識する構造
  - モデル、最適化手法、目的関数などを分けて考えられる
- 導入が簡単であること

NICEで試した結果をここに反映する

第三者がリポジトリを見たときにどこに何があるかが直感的に分かること

本質的でないコードがモデルに紛れ込まないようにする

## 使い方

新しいプロジェクトの開始

```
git clone git@github.com:nazfox/nazlab.git project-name
cd project-name
./nazlab lab init
```

nazlabを最新にアップデート

```
./nazlab lab update
```


## 依存ライブラリの追加手順

`./requirements.txt`へ依存パッケージを追記

