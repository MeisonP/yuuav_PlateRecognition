#2018/06/05
#车牌分割

path说明：
model:内包含有训练好的SVM分类模型，程序做分类事直接load SVM, 然后就不用再创建svm

image_preprocess:原始图片+输出结果
image_preprocess_segment_2:核心算法模块
	包含：分割流程yellow-->blue-->sobel
	     车牌判断 cutimage-->unisize-->SVM predicate

问题：截至0605，原始图片中的7，8，10 这三张是没有定位出来的。所以问题在于这个算法的不足，还需要进一步天调整。
