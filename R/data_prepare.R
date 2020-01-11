# �������ݼ�����
rm(list=ls())
setwd("d:\\Documents\\wuhaoyu\\����\\daguan\\data")
library(MASS)
# ����׼��
# �жϺ���
is_manager <- function(x){
  if(x=="Manager"){
    return(T)}
  else{
    return(F)
  }
}
# box-cox�任����
box_cox<-function(y,D){
  b=boxcox(y~y, data=D) # ���庯�����ͺ�����
  I=which(b$y==max(b$y))
  return(b$x[I])#lambda=0.83 
}

# ���ݼ��أ�����Ԥ����
train_Xy <- read.csv("pfm_train.csv", header = TRUE, stringsAsFactors = FALSE)
train_X <- train_Xy[-grep("Attrition", colnames(train_Xy))]
train_y <- train_Xy["Attrition"]
train_y$Attrition <- as.factor(train_y$Attrition)
test_X <- read.csv("pfm_test.csv", header = TRUE, stringsAsFactors = FALSE)
all_X <- rbind(train_X, test_X)

# ������ϴ������ɾ����Ч����
names(all_X)
names(all_X)[1] <- "Age"
all_X <- all_X[-grep("Over18|StandardHours", colnames(all_X))]

# �����������
# �Ƿ����쵼
all_X['is_manager'] <- apply(all_X['JobRole'] ,MARGIN = 1, function(x)  is_manager(x))
# ɾ����ɫ����
all_X <- all_X[-grep("JobRole", colnames(all_X))]
# ƽ����˾��ְ����
all_X["avg_per_company"] <- all_X["TotalWorkingYears"]/(all_X["NumCompaniesWorked"]+0.5)
# �����ָ��
all_X["satisfaction"] <- all_X["RelationshipSatisfaction"]*all_X["EnvironmentSatisfaction"]*all_X["JobSatisfaction"]

# ��������
num_var <- c('Age',
             'DistanceFromHome',
             'MonthlyIncome',
             'NumCompaniesWorked',
             'PercentSalaryHike',
             'TotalWorkingYears',
             'YearsAtCompany',
             'YearsInCurrentRole',
             'YearsSinceLastPromotion',
             'YearsWithCurrManager',
             'avg_per_company',
             'satisfaction'
              )
ord_var <- c('BusinessTravel',
             'Education',
             'EnvironmentSatisfaction',
             'JobInvolvement',
             'JobLevel',
             'JobSatisfaction',
             'RelationshipSatisfaction',
             'StockOptionLevel',
             'WorkLifeBalance',
             'TrainingTimesLastYear'
              )
chr_var <- c('Department',
             'EducationField',
             'EmployeeNumber',
             'Gender',
             'MaritalStatus',
             'OverTime',
             'PerformanceRating',
             'is_manager'
              )

# ����ת��
# ��������ת��
all_X[c(chr_var,ord_var)] <- lapply(all_X[c(chr_var,ord_var)], function(x) as.factor(x))
# boxcox�任
all_X[num_var] <- all_X[num_var]+0.5
lambda <- sapply(num_var, function(x) box_cox(all_X[[x,exact=FALSE]],all_X))
lambda_df <- as.data.frame(lambda)
all_X[num_var] <- lapply(num_var, function(x) all_X[[x,exact=FALSE]]^lambda_df[x,"lambda"])

# ���ݼ�����
# �������ѵ���������ֲ��Լ�
train_df_X <- all_X[which(all_X$EmployeeNumber%in%train_X$EmployeeNumber),]
train_df <- cbind(train_df_X,train_y)
train_df <- train_df[-grep("EmployeeNumber", colnames(train_df))]
index <- sample(1:nrow(train_df), nrow(train_df)*0.7, replace =F)
train_df_1 <- train_df[index,]
train_df_2 <- train_df[-index,]

# ʹ��ȫ������ѵ��ģ��
TRAIN_X <- all_X[which(all_X$EmployeeNumber%in%train_X$EmployeeNumber),]
TRAIN <- cbind(TRAIN_X,train_y)
TRAIN <- TRAIN[-grep("EmployeeNumber", colnames(TRAIN))]
TEST <- all_X[which(all_X$EmployeeNumber%in%test_X$EmployeeNumber),]
TEST <- TEST[-grep("EmployeeNumber", colnames(TEST))]
###################################################################################################
#ģ��ѵ��