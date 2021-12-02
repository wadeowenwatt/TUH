# Machine Learning Revise

## 1. Các dạng mô hình học máy theo phân loại dựa vào phương pháp thực hiện:

**Thuật toán hồi quy (Regression Algorithms)**
*Ý tưởng:* Hồi quy là quá trình tìm mối quan hệ phụ thuộc của một biến (biến phụ thuộc) vào một hoặc nhiều biến khác (biến độc lập) nhằm mục đích ước lượng hoặc tiên đoán giá trị kỳ vọng của biến phụ thuộc khi biết trước giá trị của biến độc lập.
*Ví dụ một số mô hình:*
- Linear Regression
- Logistic Regression

**Thuật toán phân loại (Classification Algorithms)**
*Ý tưởng:* 
  - Bài toán phân lớp là quá trình phân lớp một đối tượng dữ liệu vào một hay nhiều lớp đã cho trước nhờ một mô hình phân lớp (model). Mô hình này được xây dựng dựa trên một tập dữ liệu được xây dựng trước đó có gán nhãn (hay còn gọi là tập huấn luyện). Quá trình phân lớp là quá trình gán nhãn cho đối tượng dữ liệu.
  - Như vậy, nhiệm vụ của bài toán phân lớp là cần tìm một mô hình phần lớp để khi có dữ liệu mới thì có thể xác định được dữ liệu đó thuộc vào phân lớp nào.
  
*Ví dụ một số mô hình:*
- Support Vector Machine (SVM)
  
**Thuật toán mạng nơron nhân tạo (Artificial Neural Network Algorithms)**
*Ý tưởng:* Artificial Neural Network (ANN) gồm 3 thành phần chính: Input layer và Output layer chỉ gồm 1 layer , hidden layer có thể có 1 hay nhiều layer tùy vào bài toán cụ thể. ANN hoạt động theo hướng mô tả lại cách hoạt động của hệ thần kinh với các neuron được kết nối với nhau

*Ví dụ một số mô hình:*
- Perceptron
- Softmax Regression

**Thuật toán Bayes (Bayesian Algorithms)**
*Ý tưởng:* Đây là nhóm các thuật toán áp dụng Định lý Bayes cho bài toán phân loại và hồi quy.

*Ví dụ một số mô hình:*
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
  - Naive Bayes

**Thuật toán Giảm chiều dữ liệu (Dimensionality Reduction Algorithms)**
*Ý tưởng:* nói một cách đơn giản, là việc đi tìm một hàm số, hàm số này lấy đầu vào là một điểm dữ liệu ban đầu D rất lớn, và tạo ra một điểm dữ liệu mới có số chiều K < D.

*Ví dụ một số mô hình:*
-  Principal Component Analysis (PCA)

**Thuật toán dựa trên mẫu(Instance-based Algorithms)**
*Ví dụ một số mô hình:*
  - k-Nearest Neighbor (kNN)
  - Learning Vector Quantization (LVQ)
  
**Thuật toán chuẩn hoá (Regularization Algorithms)**
*Ví dụ một số mô hình:*
  - Ridge Regression
  - Least Absolute Shrinkage and Selection Operator (LASSO)
  - Least-Angle Regression (LARS)
  
**Thuật toán cây quyết định (Decision Tree Algorithms)**
*Ví dụ một số mô hình:*
- Chi-squared Automatic Interaction Detection (CHAID)
- Classification và Regression Tree (CART)

**Thuật toán phân cụm (Clustering Algorithms)**
*Ví dụ một số mô hình:*
- k-Means

**Ensemble Algorithms**
*Ví dụ một số mô hình:*
- Boosting
- AdaBoost
- Random Forest

## 2. Phương pháp đánh giá hiệu năng của một mô hình học máy

**Các tiêu chí đánh giá hiệu năng**
1. Tính chính xác (Accuracy): Mức độ dự đoán (phân lớp) chính xác của hệ thống (đã được huấn luyện) đối với các ví dụ kiểm chứng (test instances) 
2. Tính hiệu quả (Efficiency): Chi phí về thời gian và tài nguyên (bộ nhớ) cần thiết cho việc huấn luyện và kiểm thử hệ thống
3. Khả năng xử lý nhiễu (Robustness): Khả năng xử lý (chịu được) của hệ thống đối với các ví dụ nhiễu (lỗi) hoặc thiếu giá trị 
4. Khả năng mở rộng (Scalability): Hiệu năng của hệ thống (vd: tốc độ học/phân loại) thay đổi
như thế nào đối với kích thước của tập dữ liệu
5. Khả năng diễn giải (Interpretability): Mức độ dễ hiểu (đối với người sử dụng) của các kết quả và hoạt động của hệ thống

**Bài toán ví dụ**
Bài toán phân loại ảnh đó là mèo hay không, trong dữ liệu dự đoán có 100 ảnh là mèo, 1000 ảnh không phải là mèo. Ở đây, kết quả dự đoán là như sau:
Trong 100 ảnh mèo dự đoán đúng 90 ảnh, còn 10 ảnh được dự đoán là không phải.
Trong 1000 ảnh không phải mèo, dự đoán đúng được 940 ảnh không phải mèo, còn 60 ảnh bị dự đoán nhầm sang mèo

**Độ đo tính đúng đắn**
1. Accuracy (độ chính xác) 
   - Chỉ đơn giản đánh giá mô hình thường xuyên dự đoán đúng đến mức nào. 
   - Độ chính xác là tỉ lệ giữa số điểm dữ liệu được dự đoán đúng và tổng số điểm dữ liệu
   Ví dụ kết quả cho bài toán trên:
   Accuracy = (90+940)/(1000+100) = 93.6%

2. Confusion Matrix
   - Nó thể hiện được có bao nhiêu điểm dữ liệu thực sự thuộc vào một class, và được dự đoán là rơi vào một class
   - Ví dụ kết quả cho bài toán trên:
   Trong 100 ảnh mèo dự đoán đúng 90 ảnh, còn 10 ảnh được dự đoán là không phải. Nếu ta coi cat là “positive” và non-cat là “negative”, thì 90 ảnh được dự đoán là cat, được gọi là True Positive, còn 10 ảnh được dự đoán non-cat kia được gọi là False Negative
   Trong 1000 ảnh non-cat, dự đoán đúng được 940 ảnh là non-cat, được gọi là True Negative, còn 60 ảnh bị dự đoán nhầm sang cat được gọi là False Positive
3. Precision and Recall
   - Sinh ra để khắc phục nhược điểm của Accuracy
   - Precision sẽ cho chúng ta biết thực sự có bao nhiêu dự đoán Positive là thật sự True
     CT: $ Precision = \frac{True Positive}{True Positive + Fale Positive} $

   - Recall đo lường tỷ lệ dự báo chính xác các trường hợp positive trên toàn bộ các mẫu thuộc nhóm positive
     
     CT: $ Recall = \frac{True Positive}{True Positive + False Negative}$
4. F1-score
   - F1-score là kết hợp cả Recall và Precision lại được gọi là 
  CT: $ F1-Score = \frac{2*Precison*Recall}{Precison + Recall}$
5. Khác
   - Sensitivity – Specificity
   - AUC
  
## 3. Phân loại các mô hình học máy dựa vào tính chất của đầu ra

| Regression Algorithm                              | Classification Algorithm                                             |
| ------------------------------------------------- | -------------------------------------------------------------------- |
| output : biến liên tục hoặc giá trị thực          | biến đầu ra giá trị rời rạc                                          |
| ánh xạ biến đầu vào x với biến đầu ra liên tục y  | ánh xạ biến đầu vào x với biến đầu ra rời rạc y                      |
| dữ liệu liên tục                                  | dữ liệu rời rạc                                                      |
| cố tìm ra fit vừa nhất , dự đoán chính xác đầu ra | tìm đường biên(boundary) và chia tập dữ liệu thành các lớp khác nhau |
| VD : dự đoán thời tiết ,giá nhà                   | Phân loại nhận dạng email,giọng nói, tế bào ung thư,..               |
| 2 phần : Linear and Non-linear Regression         | 2 phần : Binary Classifier and Multi-class Classifier.               |


## 4. Trình bày khái niệm “Ước lượng hợp lý cực đại” (Maximum Likelikood Estimation).
**Khái niệm**
Maximum Likelikood Estimation - Ước lượng hợp lý cực đại:
- Dùng để ước lượng giá trị tham số của một mô hình xác suất
/thống kê dựa trên những dữ liệu quan sát được.
- Dựa vào việc cực đại hóa Likelihood function => Bộ tham số - phương pháp được coi là “Hợp lý cực đại”. 
- Theo Suy diễn Bayes: MLE - trường hợp đặc biệt của Maximum A Posteriori estimation (MAP), đưa ra giả thiết về phân phối đều
của các tham số
- MLE không khẳng định về xác suất của các
tham số mà chỉ khẳng định về xác suất của các ước lượng

**Công thức**
![Công thức](img/MLE1.png)
![Công thức](img/MLE2.png)
![Công thức](img/MLE3.png)

**Ví dụ sử dụng MLE**
Sử dụng trong Gaussian naive Bayes để tìm bộ tham số $\theta$ = {kỳ vọng, phương sai} dựa trên các điểm trong training set thuộc class c

## 5. Nêu các hướng tiếp cận đã học đối với các mô hình phân loại.

**Nêu các hướng tiếp cận:**
- mô hình tạo sinh
- mô hình phân biệt

**Nêu sự khác nhau giữa hai hướng tiếp cận**
1. Mô hình phân biệt: Dựa vào training data(x,y) để giải quyết phân lớp theo 2 cách:
- Học mô hình(hàm) f : x -> y = f(x)
- Với dữ liệu x, tính khả năng x có nhãn y (xác suất có điều kiện: P(y|x)
2. Mô hình tạo sinh:
- Dựa vào mô tả phân phối dữ liệu trong bản thân tập dữ liệu để xác định một dữ liệu mới
thuộc một nhãn sẽ như thế nào
- Bằng cách lấy mẫu từ mô hình này, ta có thể tạo ra dữ liệu mới.
- Mô hình này cho biết khả năng(xác suất) xảy ra của một mẫu dữ liệu mới.

**Khác nhau:** Mô hình tạo sinh thì có thể dự đoán dữ liệu sẽ có nhãn mới còn mô hình phân biệt thì chỉ dự đoán dữ liệu thuộc vào một trong các lớp có sẵn.

**Nêu ví dụ một số mô hình cho mỗi hướng tiếp cận, giải thích tại sao (ko biết)**
*Các mô hình phân biệt bao gồm:*
1. Hồi quy logistic
2. Softmax
3. K-means
4. SVM

*Mô hình tạo sinh bao gồm các phân lớp Naive Bayes*

## 6. Liên hệ giữa dữ liệu và các mô hình học máy

**Khái niệm về các loại phân bố của các tập dữ liệu (tách được/tách được tuyến tính/gần như tách được tuyến tính …)**
- Tách được tuyến tính : sử dụng dạng siêu phẳng (đường thẳng hoặc mặt phẳng)
y = f(x) = Ax + b(y,b thuộc Y thuộc R^m, x thuộc X thuộc R^n )
- Gần như tách được tuyến tính : giống tách được tuyến nhưng có 1 số điểm nằm sai lớp
không đáng kể
- Tách được : ko biết

**Trong các mô hình học máy đã học, dữ liệu thường được chia thành các tập như thế nào?**
Chia thành 3 tập : train ,validation, test

**Giải thích ý nghĩa và đặc điểm của các tập dữ liệu**
- Training dataset : tìm tham số tối ưu mô hình đó có khả năng tổng quát hóa cho dữ liệu
mới
- Validation : kiểm chứng mô hình khớp dữ liệu không để tránh overfit
- Testing data : Dữ liệu cần kiểm tra chưa được gán nhãn

**Cho biết các tập dữ liệu nào sẽ được sử dụng trong các dạng phương pháp học máy nào?**
Train ,val,test dùng cho học có giám sát

**Nêu sự khác nhau về đặc điểm của tập dữ liệu Test và các tập còn lại**

-Train : Mẫu dữ liệu được sử dụng để cung cấp đánh giá không thiên vị về sự phù
hợp của mô hình trên tập dữ liệu huấn luyện trong khi điều chỉnh các siêu tham số của
mô hình.
-Validation : Mẫu dữ liệu được sử dụng để cung cấp đánh giá không thiên vị về sự
phù hợp của mô hình cuối cùng trên tập dữ liệu huấn luyện

**Nêu sự khác nhau về mục đích sử dụng của Validation Set và Training Set**

- Mục đích Training: Mục tiêu là tạo ra một mô hình được đào tạo (được trang bị) có
khả năng tổng quát hóa tốt cho dữ liệu mới, chưa biết. Mô hình được trang bị được
đánh giá bằng cách sử dụng các dữ liệu “mới” từ các bộ dữ liệu đã tổ chức (bộ dữ
liệu train và validation ) để ước tính độ chính xác của mô hình trong việc phân loại
dữ liệu mới.
- Mục đích Validation: Kiểm chứng mô hình khớp dữ liệu không để tránh overfit

## 8. Mô hình hồi quy tuyến tính (Linear Regression)
### ***a, Đặc điểm khác biệt của mô hình hồi quy tuyến tính so với các mô hình phân loại*** 
```
    Mô hình hồi quy tuyến tính sử 
```
### ***b, Hướng tiếp cận của mô hình hồi quy tuyến tính***
 ```
Mô hình hồi quy tuyến tính tiếp cận theo hướng phân loại phân biệt 
 ``` 
### ***c, Cách xây dựng hàm tổn thất và căn cứ để xác định tham số***
 * Hàm tổn thất được xây dựng dựa trên công thức bình phương tối thiểu sai số giữa đầu ra thực tế và đầu ra dự đoán, với đầu ra ở đây là các **_label_** (nhãn dữ liệu)
 * với tất cả các cặp (input, output) (xi, yi), i = 1, 2, . . . , N, với N là số lượng dữ liệu quan sát được. Điều chúng ta mong muốn–trung bình sai số là nhỏ nhất–tươngđương với việc tìm **w** để hàm số sau đạt giá trị nhỏ nhất:
    
    $$ 
        L(w) = \frac{1}{2N} \sum_{i=1}^{N} {(y_i - x^T_iw)}^2
    $$
 * Hàm số $L(w)$ chính là hàm mất mát của _Linear Regression_ với tham số mô hình $\theta$ = **w**. Chúng ta mong muốn sự mất mát là nhỏ nhất, điều này có thể đạt được bằng cách tối thiểu hàm mất mát theo **w**:
  
    ```
        $w^* = \argmin_w L(w) (8.1)$
    ```

 * Để xác định tham số cho bài toán chúng ta đi giải bài toán tối ưu $(8.1)$ bên trên. Nhận thấy rằng hàm mất mát $L(w)$ có đạo hàm tại mọi **w**. Việc tìm giá trị tối ưu **w** có thể được thực hiện thông qua việc giải phương trình đạo hàm $L(w)$ theo **w** bằng 0:
    $$`
        \frac{\partial{L(w)}}{\partial{w}} = \frac{1}{N}X(X^Tw- y)
    $$
 * Giải phương trình đạo hàm bằng 0:
    $$
    \frac{\partial{L(w)}}{\partial{w}} = 0 
    $$
    $$
        XX^Tw = Xy (8.2)
    $$
 * Nếu ma trận $XX^T$ khả nghịch phương trình $(8.2)$ có nghiệm duy nhất $w=(XX^T)^{-1}Xy$
 * Bias trick  (hệ số tự do):
  
    ![Bias Trick](./img/bias_linear_regression.png)

 * Trường hợp $XX^T$ không khả nghịch:
 
    ![Không khả nghịch](./img/irreversibility.png)

## 9. Mô hình hồi quy Logistic:
### **_a,Phân kiểu của phương pháp (theo phương pháp/theo dạng bài toán/theo hướng tiếp cận)_** 

### **_b. Loại dữ liệu có thể áp dụng_**
    
### **_c. Hàm kích hoạt logistic (sigmoid): đặc điểm và tại sao được sử dụng trong phương pháp_**

### **_d. Cách xây dựng hàm tổn thất và phương pháp giải bài toán tối ưu hình thành từ đó._**

