from mnist import MNIST
from sklearn import linear_model

mndata = MNIST('../mnist')
images, labels = mndata.load_training()

lasso = linear_model.LogisticRegression(penalty='l1',
                                      multi_class='multinomial',
                                      solver='saga', tol=0.1)
lasso.fit(images, labels)
