# Laporan Proyek Machine Learning - Mawan
🌀 #6. "Machine Learning Terapan - Predictive Analytics"

<br>

<p align='center'>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />        
  </a>
  <a href="https://github.com/nurmuhimawann/">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />        
  </a>
  <a href="https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" />
  </a
   <a href="https://colab.research.google.com/">
    <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" />
  </a
</p>

<br>

| Profile       |                                           |
| ------------- | ----------------------------------------- |
| Nama          | Nur Muhammad Himawan                      |
| Learning Path | Machine Learning & Front End Development  |
| Progam        | Studi Independen Batch 3 - Kampus Merdeka |
| Submission    | Proyek Machine Learning Terapan           |



<p align='center'>
    <img src ="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/hero-image.png?raw=true" alt="anemia">
</p>




## Domain Proyek

Domain yang akan dibahas pada proyek machine learning (*predictive analytics*) ini adalah **Kesehatan** dengan judul **"Deteksi Penyakit Anemia Berdasarkan Kondisi Sel Darah Merah"**.

### Latar Belakang

Anemia merupakan suatu keadaan dimana kadar hemoglobin dibawah normal. Hemoglobin adalah protein kaya zat besi yang memberikan warna merah pada darah. Protein ini membantu sel-sel darah merah membawa oksigen dari paru-paru ke seluruh tubuh. Sehingga apabila penyakit anemia ini dibiarkan secara berlarut-larut akan mengganggu fungsi organ-organ penting tubuh dan menimbulkan berbagai macam penyakit. Gejala yang terjadi pada penderita anemia adalah cepat lesu, lemah, letih, lelah, dan cepat lupa. Angka kejadian anemia di Indonesia terbilang masih cukup tinggi. Berdasarkan Riset Kesehatan Dasar (Riskesdas) 2018, prevalensi anemia pada remaja sebesar 32 %, artinya 3-4 dari 10 remaja menderita anemia. Namun, banyak orang tidak menyadari hal ini. Sehingga, apabila penderita anemia terlambat mengetahui kondisinya, maka akan terjadi dampak yang lebih serius apabila tidak segera ditangani. Dari permasalahan tersebut, dapat dilihat bahwa perlu adanya sistem predictive untuk mendeteksi penyakit anemia ini.

Ada banyak sistem predictive yang berkembang, salah satunya dengan pendekatan klasifikasi. Klasifikasi merupakan proses pencarian sekumpulan model atau fungsi yang menggambarkan dan membedakan kelas data. Tujuan dari klasifikasi adalah untuk memprediksi kelas dari suatu objek yang belum diketahui kelasnya. Klasifikasi sendiri memiliki dua proses yaitu membangun model klasifikasi dari sekumpulan kelas data yang sudah didefinisikan sebelumnya (training data) dan menggunakan model tersebut untuk klasifikasi data test serta mengukur akurasi dari model (testing data). Dalam prosesnya, klasifikasi memiliki empat komponen, yaitu kelas merupakan variabel terikat yang merepresentasikan label yang terdapat pada objek. kedua, predictor adalah variabel bebas yang direpresentasikan oleh atribut. ketiga, data training adalah set data yang digunakan untuk menentukan kelas yang cocok berdasarkan predictor. terakhir, data test adalah set data baru yang akan diklasifikasikan oleh model yang telah dibuat sebelumnya. Pada proyek machine learning ini, klasifikasi akan digunakan untuk membuat model machine learning.

Referensi:

- [Classification of Anemia with Digital Images of Nails and Palms using the Naive Bayes Method](http://jurnal.upnyk.ac.id/index.php/telematika/article/view/4587) 
- [Remaja Sehat Komponen Utama Pembangunan SDM Indonesia](https://www.kemkes.go.id/article/view/21012600002/remaja-sehat-komponen-utama-pembangunan-sdm-indonesia.html) 
- [Data Mining Techniques: Types of Data, Methods, Applications](https://www.upgrad.com/blog/data-mining-techniques/) 

## Business Understanding

### Problem Statements

Berdasarkan uraian yang telah dipaparkan pada latar belakang diatas, maka dapat diambil sebuah rumusan masalah yang dirumuskan sebagai berikut:
- Bagaimana memprediksi penyakit anemia pada manusia berdasarkan data sel darah merah dengan menerapkan algoritma machine learning?
- Bagaimana mengembangkan model machine learning yang dapat memprediksi penyakit anemia pada manusia berdasarkan data sel darah merah?
- Bagaimana evaluasi model machine learning yang dapat memprediksi penyakit anemia pada manusia berdasarkan data sel darah merah?

### Goals

Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka proyek penelitian ini memiliki tujuan, yaitu:
- Mengetahui penerapan algoritma machine learning dalam memprediksi penyakit anemia pada manusia berdasarkan data sel darah merah.
- Mengetahui proses pengembangan model machine learning dalam memprediksi penyakit anemia pada manusia berdasarkan data sel darah merah.
- Mengetahui hasil evaluasi model machine learning dalam memprediksi penyakit anemia pada manusia berdasarkan data sel darah merah.

### Solution statements

Berdasarkan tujuan yang telah dipaparkan diatas, maka proyek penelitian ini memiliki solusi atau tahapan sebagai berikut:

- Melakukan data preprocessing dan exploratory data analysis pada dataset yang dipakai dengan memvisualiasasikan keterkaitan antar feature pada data tersebut untuk mendapatkan insight dan knowledge.
- Mengembangkan model menggunakan algoritma klasifikasi yang sesuai dengan memprediksi kelas atau kategori data dengan memanfaatkan nilai yang ada pada dataset. Untuk mencapai keakuratan hasil prediksi, apabila nilai metriks yang didapatkan pada evaluation model kurang memuaskan, maka akan dilakukan improvement pada baseline model pada masing-masing algoritma menggunakan hyperparameter tuning. Algoritma yang akan dipakai diantaranya adalah sebagai berikut:
  - K-Nearest Neighbors (KNN), merupakan algoritma sederhana yang dapat mengklasifikasikan data berdasarkan tetangga terdekatnya sebagai acuan menggunakan prinsip similarity.
  - Support Vector Machine (SVM), merupakan algoritma yang dapat digunakan untuk menyelesaikan permasalahan klasifikasi dengan menemukan *hyperplane* yang optimal dalam ruang n-dimensi.
  - Random Forest, merupakan algoritma *machine learning* berbasis ensemble yang dapat digunakan pada permasalahan klasifikasi. random forest berkerja dengan sekumpulan tree untuk menghasilkan prediksi yang baik.
- Melakukan evaluation model menggunakan metriks evaluasi. evaluasi ini dilakukan untuk mengetahui kinerja atau performance dari model yang telah dikembangkan.

## Data Understanding
Dataset yang dipakai dalam proyek machine learning ini merupakan dataset anemia dengan 1421 records data. Dataset ini bersifat open-source yang dipubilkasikan oleh Biswa Ranjan Rao melalui platform [Kaggle: Anemia Dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset). Topik dari datasetnya adalah kesehatan yang berformat csv (comma-separated values) dengan ukuran 34.63 kB.

<p align='center'>
    <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/header_dataset.png?raw=true" alt="datasets">
</p>

### Variabel-variabel pada Anemia dataset:
- Gender : jenis kelamin (0 = Laki-laki, 1 = Perempuan).
- MCH : *Mean Corpuscular Hemoglobin* merupakan jumlah rata-rata hemoglobin di dalam sel darah merah.
- MCHC : *Mean Corpuscular Hemoglobin Concentration*, konsentrasi rata-rata hemoglobin dalam setiap sel darah merah.
- MCV : *Mean Corpuscular Volume*, perhitungan ukuran rata-rata sel darah merah.
- Results : Status ada tidaknya penyakit anemia (0 = Tidak, 1 = Iya)

### Visualization & Analysis

- **Univariate Analysis**

<p align='center'>
    <img src ="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/pie-chart.png?raw=true" height=350px alt="pie-chart">
</p>

Berdasarkan visualiasi pie-chart diatas, orang yang mengidap penyakit anemia lebih sedikit daripada orang yang tidak mengidap anemia, yakni sebesar 46.3%. Namun, perlu diketahui bahwa angka ini cukuplah besar mengingat hal ini berkaitan langsung dengan kondisi kesehatan.

- **Bivariate Analysis**

<p align='center'>
    <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/hb.corr.png?raw=true" height=350px alt="hb-corr">
</p>

Dengan menggunakan boxplot, distribusi data dapat terlihat dengan jelas terkait korelasi antara feature gender dan hemoglobin. dari sini, insight yang didapatkan adalah orang yang terkena anemia, memiliki kecenderungan pada kadar hemoglobin yang rendah. Jadi, rendahnya kadar hemoglobin dapat memberikan indikasi kalau terkena penyakit anemia.

<p align='center'>
    <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/mch.mcv.mchc.png?raw=true" height=auto alt="mch.mcv.mchc">
</p>

Pada feature MCV, MCH dan MCHC, tidak ada insight baru yang ditemukan. karena, tidak ada pola atau perbedaan yang secara signifikan dapat terlihat. 

**Multivariate Analysis**

<p align='center'>
    <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/pairplot.png?raw=true" height=350px alt="pairplot">
</p>

Pada multivariate analysis melalui pairplot, dapat terlihat jelas kalau tidak ada keterkaitan yang jelas dengan variabel result selain dengan hemoglobin. 

**Outlier & Distribution Analysis**

<p align='center'>
    <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/boxplot-outlier.png?raw=true" height=auto alt="boxplots-outlier">
</p>

<p align='center'>
    <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/histplot.png?raw=true" height=auto alt="hist">
</p>

Visualiasi boxplot dapat membantu untuk mengidentifikasi ada tidaknya outlier data pada masing-masing feature. dapat dilihat kalau pada data yang dipakai tidak memiliki outlier. kemudian, pada histogram distribusi normal dapat terlihat pada feature MCH, MCHC, dan MCV. sedangkan, feature Hemoglobin memiliki kecenderungan sedikit miring ke kiri (left-skewed).

## Data Preparation
Teknik yang digunakan dalam Data Preparation adalah sebagai berikut:

- One-Hot Encoding

  Ini merupakan metode yang sangat populer digunakan untuk mengubah data kategorikal ke data numerical dengan nilai biner 0 atau 1. Proses encoding sangat diperlukan, agar data yang masuk ke dalam algoritma machine learning dapat bekerja dengan baik. Sebagian besar algoritma klasifikasi lebih mudah untuk memproses nilai numerical daripada kategorikal. Pada tahap ini, encoding dilakukan secara manual dengan membuat function untuk mengubah values data kategorikal ke numerical. Hal ini mudah untuk dilakukan secara manual karena saya telah memahami dan mengetahui values dari feature nya. Feature yang perlu untuk dilakukan encoding adalah `Gender` dan `Result`. 

- Data Splitting

  Proses ini merupakan tahap untuk membagi dataset menjadi data train dan test. Pembagian ini bertujuan agar data yang digunakan dapat digunakan untuk mengembangkan model dan mengevaluasi performance dari model yang sudah dikembangkan. Pada proyek ini, dataset akan di split dengan proporsi 80% atau 427 data untuk data train dan 20% atau 107 data untuk data testing. Sedangkan, proses splitting akan menggunakan function train_test_split() yang tersedia pada library sklearn.

- Feature Scaling (Standarisasi)

  Scaling bertujuan untuk menormalisasikan range pada fitur - fitur data agar seluruh fitur berada pada range yang sama. Apabila model machine learning tidak melakukan proses feature scaling, maka hasil prediksi akan lebih condong atau didominasi oleh fitur yang memiliki values terbesar, sementara fitur dengan values yang kecil, memiliki peluang yang kecil untuk mempengaruhi hasil prediksi. Pada proyek ini, scaling data akan menggunakan metode standarisasi. karena secara umum distribusi data berada pada kondisi normal dan standarisasi lebih cocok untuk digunakan dalam case yang seperti ini. proses standarisasi akan dilakukan dengan memanfaatkan function StandardScaler() yang tersedia di dalam library sklearn. Proses standarisasi bekerja dengan mengurangkan data pada fitur dengan nilai rata-rata fitur (mean), yang kemudian dibagi dengan standar deviasi. proses ini akan mengasumsikan semua fiturnya terpusat disekitaran nol dan memiliki besaran variansi yang sama.

- Handling Imbalanced Class

  Proporsi dari label/class pada dataset yang tidak seimbang akan menjadi permasalahan yang cukup besar, khususnya pada algoritma klasifikasi. Hal ini dikarenakan, algoritma machine learning yang digunakan akan cenderung mengklasifikasikan data ke dalam class yang memiliki data yang lebih banyak atau dominan (majority class) daripada kelas yang lebih sedikit (minority class). Hal ini akan sangat berbahaya apabila terjadi, khususnya dalam bidang kesehatan. dimana kesalahan hasil prediksi bisa saja berakibat fatal bagi pasien. Pada dataset yang dipakai dalam proyek ini, terdapat imbalanced class. dimana Synthetic Minority Over-sampling Technique (SMOTE) akan dipakai untuk handling pada kasus ini. SMOTE memakai pendekatan oversampling, hal ini dilakukan dengan mensintesis sampel baru dari minority class untuk menyeimbangkan dataset dengan cara membuat instance baru dari *minority* *class*. Dengan metode ini dapat membuat data-set menjadi seimbang.
  
  <p align='center'>
      <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/handling-imbalanced.png?raw=true" width=600px alt="smote">
  </p>

## Modeling
Pada proyek ini, algoritma *machine learning* yang dipakai adalah `K-Nearest Neighbor`, `Support Vector Machine`, `Logistic Regression` dan `Random Forest`.

- **K-Nearest Neighbor (KNN)**

  Algoritma KNN atau K-Nearest Neighbor merupakan salah satu algoritma paling sederhana dan populer digunakan dalam klasifikasi pada machine learning. KNN bekerja dengan mengambil sejumlah K-data untuk dijadikan acuan dalam menentukan class dari data yang baru. Setiap data akan dibandingkan berdasarkan jarak (similarity) antara satu data dengan data lainnya dengan memilih K tetangga terdekat. Proses modelling menggunakan KNN dalam proyek ini, akan memakai modul yang telah tersedia pada library scikit-learn yakni [KNeighborsClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) dengan parameters `n_neighbors = 8` yang artinya akan ada 8 data acuan yang akan digunakan sebagai K tetangga terdekat dalam proses klasifikasi. kemudian, metrics yang digunakan dalam menentukan similarity adalah `minkowski distance`. cara kerja minkowski hampir mirip dengan euclidian distance, hanya saja yang membedakan adalah penambahan parameter p atau pangkatnya. minkowski menghitung jarak antar 2 vektor data. apabila nilai p=1 maka itu adalah manhattan, sedangan p=2 itu adalah euclidian distance. berikut formula dari minkowski distance. `d(x-y)=(∑i=1n|xi−yi|p)1/p.`

  <p align='center'>
      <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/knn.png?raw=true" height=300px alt="knn">
  </p>

  Proses building model dalam KNN akan terus dilakukan iterasi secara berulang dengan mencari nilai K tetangga terdekat sampai mendapatkan hasil yang optimal. setelah model didapatkan, tahap selanjutnya akan dilakukan proses testing menggunakan data test yang telah disediakan.

  - Kelebihan:

    Algoritma KNN merupakan algoritma yang sederhana dan mudah untuk diimplementasikan.

    Dapat di implementasikan pada beberapa kasus seperti klasifikasi, regresi dan pencarian.

  - Kekurangan:

    Algoritma KNN menjadi lebih lambat secara signifikan seiring meningkatnya jumlah sampel dan/atau variabel independen.

- **Support Vector Machine (SVM)**

  Algoritma SVM bekerja untuk menemukan hyperplane atau pemisah yang dapat memaksimalkan jarak (margin) antar kelas dalam ruang n-dimensi untuk mengklasifikasikan titik-titik data. Memaksimalkan jarak margin akan memberikan kejelasan terkait klasifikasi kelas sehingga titik data yang baru dilihat dapat diklasifikasikan dengan lebih baik.

  <p align='center'>
      <img src ="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/SVM.png?raw=true" alt="svm">
  </p>


  Pada tahap modelling, [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) yang dipakai menggunakan metode kernel dan menerima semua vektor input yang diberikan pada data training dengan menerapkan parameter `rbf` yang dipakai sebagai kernel tricks nya. Kernel ini dikenal memiliki performa yang baik dan hasil dari pelatihan memiliki nilai error yang kecil. fungsi kernel rbf adalah sebagai berikut `K(x,xi) = exp(-gamma \* sum((x – xi^2))`

  - Kelebihan

    Mampu bekerja dengan baik pada data yang relatif sedikit.

    Pengklasifikasian SVM dapat memberikan model dengan akurasi tinggi dan bekerja dengan baik dengan ruang dimensi tinggi.

  - Kekurangan

    Sulit diaplikasikan pada data yang sangat besar karena memiliki waktu pelatihan yang tinggi.

- **Random Forest**

  Random Forest merupakan salah satu algoritma machine learning terbaik yang digunakan dalam klasifikasi dalam jumlah data yang besar. Random Forest memakai pendekatan kombinasi dari beberapa pohon keputusan (decision tree) yang datanya akan dipilih secara random. Dalam random forest, penentuan klasifikasi dilakukan berdasarkan hasil voting dari tree yang terbentuk. sehingga, pemakaian jumlah tree yang lebih banyak dapat menghasilkan tingkat akurasi yang lebih optimal. Tree yang dihasilkan oleh random forest dilatih menggunakan metode bagging. Bagging akan bekerja dengan memilih fitur secara random dengan menerapkan sampling with replacement. Kemudian, dari hasil ini akan diperoleh model tree klasifikasi. proses ini akan terus berulang hingga mendapatkan jumlah tree (k) yang diinginkan. kemudian dari jumlah tree yang ada, masing-masing tree akan memberikan hasil prediksi. langkah terakhir, proses *majority voting* akan dilakukan untuk menentukan prediksi akhir. Pada proyek machine learning ini, implementasi random forest akan dilakukan dengan memakai modul [RandomForestClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) yang telah tersedia pada library scikit-learn. parameter `n_estimator` dipakai untuk menentukan jumlah tree. disini saya memakai 100 tree. Kemudian setelah menentukan parameter model, proses selanjutnya adalah building model dan prediksi yang dilakukan menggunakan data testing. hasil dari testing akan dievaluasi menggunakan metriks accuracy.

  <p align='center'>
      <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/random-forest.png?raw=true" height=300px alt="random-forest">
  </p>
  
  - Kelebihan :
    
    Random Forest bekerja sangat baik pada data dengan jumlah yang sangat besar.
    
    Hasil pembelajaran yang diperoleh pada random forest memiliki tingkat akurasi yang sangat baik.
    
    Random Forest dapat memberikan perkiraan variabel yang penting dalam proses klasifikasi.
    
  - Kekurangan :
    
    Untuk type data kategorikal, random forest tidak bisa bekerja dengan optimal dan cenderung menghasilkan hasil prediksi yang bias.
    
    Waktu runtime yang lama karena random forest menggunakan data dalam jumlah yang besar dan random tree yang banyak pula.

#### **Pemilihan Model**: 

Dari ketiga model yang telah dilatih, Random Forest merupakan model terbaik yang dihasilkan. Hal ini didasarkan pada confusion matrix yang didapatkan dari hasil testing secara sempurna yakni 100% pada confusion-matrix. Secara umum, 2 model lainnya juga memberikan hasil yang cukup bagus. Sehingga proses improvement dengan hyperparameter tuning tidak perlu untuk dilakukan, karena model yang dikembangkan sudah memenuhi ekspetasi dari solution statement yang sudah di tentukan di awal.

<p align='center'>
    <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/conf-matrix.png?raw=true" height=auto alt="conf-matrix">
</p>



## Evaluation
Berdasarkan pada kasus yang akan diselesaikan, yakni klasifikasi. maka ada empat metrik evaluasi yang populer digunakan yaitu **akurasi, precision, recall, dan F1 score**.

- accuracy, metriks yang mengukur rasio data yang diprediksi dengan benar terhadap total sample.
- precision, metriks yang mengukur tentang seberapa tepat/akurat model yang dikembangkan dari data yang diprediksi positif. precision sangat baik, apabila digunakan dalam kondisi ketika false positive sangat tinggi. seperti deteksi spam email, apabila precision tidak tinggi, maka pengguna bisa saja kehilangan email yang penting.
- recall, metriks yang menghitung seberapa banyak actual positive yang dapat ditangkap oleh model. metriks recall akan sangat baik digunakan dalam pemilihan model apabila false negative sangat tinggi. misalnya dalam fraud detection, apabila transaksi yang curang (actual positive) diprediksi tidak curang (predicted negative) maka bank dapat mengalami kerugian yang sangat besar.
- F1 score, merupakan metriks yang digunakan untuk mencari keseimbangan antara precision dan recall.

Berdasarkan konteks data, problem statement, dan solusi yang diimplementasikan, metrik evaluasi yang digunakan pada model machine learning ini adalah recall. pemilihan metrik ini sangat cocok untuk diterapkan, khususnya untuk deteksi penyakit. metriks yang menghitung seberapa banyak actual positive yang dapat ditangkap oleh model. Apabila pasien yang menderita anemia (actual positive) menjalani tes dan diprediksi tidak sakit (predicted negative). biaya yang terkait dengan false negative akan sangat tinggi apabila penyakitnya tidak segera ditangani. artinya akan jauh lebih fatal bila model memprediksi pasien yang tidak terkena anemia padahal kenyataanya dia terkena anemia. Jadi, model yang dipilih harus mempunyai nilai recall yang tinggi.

Recall bekerja dengan membagi nilai true positive dengan penjumlahan antara nilai true positive dan false negative `Recall = TP/(TP+FN)`. Recall yang ideal harus 1 (tinggi). Recall menjadi 1 apabila pembilang dan penyebutnya sama yaitu `TP = TP +FN` , artinya `FN` nya adalah 0. Ketika `FN` meningkat, maka nilai penyebut menjadi lebih besar daripada pembilang. konsekuensinya nilai recall akan menurun.

Keterangan:

- *True Positive* (TP) : model memprediksi nilai True dan aktualnya memang benar (positive).
- *True Negative* (TN): model memprediksi nilai True tetapi aktualnya adalah negative (salah).
- *False Positive* (FP) : model memprediksi nilai positive dan jawaban yang benar adalah negative.
- *False Negative* (FN): model memprediksi nilai negative tetapi jawaban yang benar adalah positive.

<p align='center'>
    <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/recall.png?raw=true" height=auto alt="recall">
</p>

Dalam proyek machine learning ini. model terbaik yang dikembangkan sesuai case menggunakan ketiga algoritma tersebut adalah **Random Forest**. Random Forest menghasilkan nilai recall tertinggi dari ketiga algoritma yang telah diterapkan yakni 1.0.



## Daftar Pustaka

- Kemkes.go.id. (2021). “Remaja Sehat Komponen Utama Pembangunan SDM Indonesia”. Diakses pada 15 September 2022, pukul 08.00 WIB, melalui website https://www.kemkes.go.id/article/view/21012600002/remaja-sehat-komponen-utama-pembangunan-sdm-indonesia.html
- Peksi, Nandha, Bambang Yuwono, & Mangaras Yanu Florestiyanto. "Classification of Anemia with Digital Images of Nails and Palms using the Naive Bayes Method." *Telematika : Jurnal Informatika dan Teknologi Informasi* [Online], 18.1 (2021): 118-130. Web. 15 Sep. 2022
- Ramageri, B. M. (2010). Data Mining Techniques and Applications. Indian Journal of Computer Science and Engineering. Vol. 1 No. 4: 301-302
