# 基础数据集操作
rm(list=ls())
setwd("d:\\Documents\\wuhaoyu\\桌面\\daguan\\data")
library(MASS)
# 函数准备
# 判断函数
is_manager <- function(x){
  if(x=="Manager"){
    return(T)}
  else{
    return(F)
  }
}
# box-cox变换函数
box_cox<-function(y,D){
  b=boxcox(y~y, data=D) # 定义函数类型和数据
  I=which(b$y==max(b$y))
  return(b$x[I])#lambda=0.83 
}

# 数据加载，基础预处理
train_Xy <- read.csv("pfm_train.csv", header = TRUE, stringsAsFactors = FALSE)
train_X <- train_Xy[-grep("Attrition", colnames(train_Xy))]
train_y <- train_Xy["Attrition"]
train_y$Attrition <- as.factor(train_y$Attrition)
test_X <- read.csv("pfm_test.csv", header = TRUE, stringsAsFactors = FALSE)
all_X <- rbind(train_X, test_X)

# 数据清洗列名，删除无效变量
names(all_X)
names(all_X)[1] <- "Age"
all_X <- all_X[-grep("Over18|StandardHours", colnames(all_X))]

# 新增组合特征
# 是否是领导
all_X['is_manager'] <- apply(all_X['JobRole'] ,MARGIN = 1, function(x)  is_manager(x))
# 删除角色变量
all_X <- all_X[-grep("JobRole", colnames(all_X))]
# 平均公司在职年数
all_X["avg_per_company"] <- all_X["TotalWorkingYears"]/(all_X["NumCompaniesWorked"]+0.5)
# 满意度指数
all_X["satisfaction"] <- all_X["RelationshipSatisfaction"]*all_X["EnvironmentSatisfaction"]*all_X["JobSatisfaction"]

# 变量分类
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

# 特征转换
# 特征类型转换
all_X[c(chr_var,ord_var)] <- lapply(all_X[c(chr_var,ord_var)], function(x) as.factor(x))
# boxcox变换
all_X[num_var] <- all_X[num_var]+0.5
lambda <- sapply(num_var, function(x) box_cox(all_X[[x,exact=FALSE]],all_X))
lambda_df <- as.data.frame(lambda)
all_X[num_var] <- lapply(num_var, function(x) all_X[[x,exact=FALSE]]^lambda_df[x,"lambda"])

# 数据集划分
# 重新组合训练集，划分测试集
train_df_X <- all_X[which(all_X$EmployeeNumber%in%train_X$EmployeeNumber),]
train_df <- cbind(train_df_X,train_y)
train_df <- train_df[-grep("EmployeeNumber", colnames(train_df))]
index <- sample(1:nrow(train_df), nrow(train_df)*0.7, replace =F)
train_df_1 <- train_df[index,]
train_df_2 <- train_df[-index,]

# 使用全量数据训练模型
TRAIN_X <- all_X[which(all_X$EmployeeNumber%in%train_X$EmployeeNumber),]
TRAIN <- cbind(TRAIN_X,train_y)
TRAIN <- TRAIN[-grep("EmployeeNumber", colnames(TRAIN))]
TEST <- all_X[which(all_X$EmployeeNumber%in%test_X$EmployeeNumber),]
TEST <- TEST[-grep("EmployeeNumber", colnames(TEST))]
###################################################################################################
#模型训练