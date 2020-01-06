# pytorch_face_gradcam
## 概要
* 自分の顔が誰に似ているのかをpytorchのクラス分類を使って調べる事ができるスクリプトです。
* また、Grad-Camによってどのパーツを重視しているかを調べることも出来ます。
* 詳しくはこのqiita記事([[PyTorch] Grad-CAMで自分の顔が誰に似ているかを調べてみた](https://qiita.com/takanosuke/items/c22c6022a4a6d4fe2ab0))を参照してください。

## 環境
* pytorch == 1.1.0
* torchvision == 0.3.0
* facenet-pytorch == 1.0.2
* opencv-python == 4.1.2.30
* matplotlib == 3.1.0

## 詳細
### face_cropper.py
* imagesフォルダにある画像から、顔だけを切り抜いてface_cropped_imagesフォルダへと出力する
* 実行コマンド
    * `python face_cropper.py`
* 切り抜く際に自動でリサイズされる
* jpgのみに対応している

### train.py
* datasetsフォルダにある画像を使って学習し、モデルを作成する
* datasetsフォルダにある画像は、4:1でtrainデータとvalidationデータに分けられる
* 実行コマンド
    * `python train.py <引数>`
* 引数
    * `--epoch` ：エポック数を指定する(default: 100)
    * `--batch_size` ：trainデータのバッチサイズ指定する(default: 32)
    * `--val_batch_size`：validationデータのバッチサイズ指定する(default: 8)
    * `--out_weight_path`：モデルの出力先のパスを設定する(default: ./weights)
    * `--save_better_only`：Trueにするとvalidationデータの精度が上がった時のみモデルを保存する(default: True)

### datasetsフォルダ
* 学習に使う画像データを入れるフォルダ
* デフォルトではclass_1, class_2, class_3, class_4の4クラスに分類する
* 自分が分類したいクラスによって、datasets以下のフォルダの名前や数を変更する

### predict.ipynb
* 学習済みモデルを使って、画像を予測する
* get_configメソッド内の設定を確認する
    * `dataset`　：予測する画像が入ったフォルダのパスを指定する
    * `weight`：使用する学習済みモデルのパスを指定する
    * `classes`：分類したいクラスを指定する(アルファベット順に並べること)
* jupyter notebookを用いて実行する

### grad_cam.ipynb
* 学習済みモデルを使って、画像の予測の根拠を示す
* get_configメソッド内の設定を確認する
    * `dataset`　：予測する画像が入ったフォルダのパスを指定する
    * `weight`：使用する学習済みモデルのパスを指定する
    * `classes`：分類したいクラスを指定する(アルファベット順に並べること)
* jupyter notebookを用いて実行する

### augument.py
* 実行するとdatasetsフォルダ内の画像を、50%の確率でx軸反転させて増やす
* 実行コマンド
    * `python augument.py`
* データ増強が必要なければ使用する必要はない