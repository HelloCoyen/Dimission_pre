rm(list=ls())
library(pROC)
library(psych)
library(car)
set.seed(1234)
setwd("C:\\Users\\wuhaoyu\\Desktop\\DG")
# 数据加载，基础预处理
data <- read.csv("pfm_train.csv", header = TRUE)
names(data)[1] <- "Age"
data <- data[, -c(8,18,23)]
data$Attrition <- as.factor(data$Attrition)
# data$BusinessTravel[data$BusinessTravel=="Non-Travel"] <- 0
# data$BusinessTravel[data$BusinessTravel=="Travel_Rarely"] <- 1
# data$BusinessTravel[data$BusinessTravel=="Travel_Frequently"] <- 2
# data$BusinessTravel <- as.integer(data$BusinessTravel)
# View(data)
# 11个数值变量
num_var <- c("Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked", 
             "PercentSalaryHike", "StockOptionLevel", "TotalWorkingYears", "YearsAtCompany",
             "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager")
# 6个类别变量
char_var <- c("Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime")
# 10个顺序变量
ord_var <- c("BusinessTravel", "Education", "EnvironmentSatisfaction", "JobInvolvement",
             "JobLevel", "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction",
             "TrainingTimesLastYear", "WorkLifeBalance")
data[,ord_var] <- lapply(data[,ord_var], function (x) as.factor(x))
# describe(data)
# str(data)

# num_df <- data[,which(names(data)%in%num_var)]
# describe(num_df)
# cor(num_df)
# 
# char_df <- data[,which(names(data)%in%char_var)]
# 
# ord_df <- data[,which(names(data)%in%ord_var)]
# cor(ord_df, method = c("pearson", "kendall", "spearman"))

#############################################################
data$MonthlyIncome <- log(data$MonthlyIncome)
# data$YearsAtCompany <- log(data$YearsAtCompany+1)
# data$YearsSinceLastPromotion <- log(data$YearsSinceLastPromotion+1)
# 选取训练集
index <- sample(1:nrow(data), nrow(data), replace =F)
train <- data[index,]
test <- data[-index,]
# 增加1类样本
# opt_sam = data[which(data$Attrition==1),]
# opt_sam_idex = sample(1:nrow(opt_sam),nrow(opt_sam)*5,replace=T)
# opt_append = opt_sam[opt_sam_idex,]
# neg_sam = data[which(data$Attrition==0),]
# neg_sam_idex = sample(1:nrow(neg_sam),nrow(neg_sam)*5,replace=T)
# neg_append = neg_sam[neg_sam_idex,]
# train = rbind(train,opt_append, neg_append)
# 全模型
model_total <- glm(Attrition~.,
                   data = train,
                   family = binomial(link = "logit")
)
# summary(model_total)
model_total.step <- step(model_total, direction="both")
summary(model_total.step)
# 使用测试集去预测模型
pred <- predict(model_total.step, test, type='response')
res <- ifelse(pred>0.5,1,0)
# 模型的精度
accuracy <- table(res,test$Attrition, dnn = c("预测值","真实值"));accuracy
accuracy <- mean(res==test$Attrition);accuracy
#做出roc的图像
roc <- roc(test$Attrition,pred);roc
plot(roc)
######################################################################################
# 人工选择
model <- glm(formula = Attrition ~ Age + BusinessTravel + Department + 
               DistanceFromHome + EducationField + EnvironmentSatisfaction + 
               Gender + JobInvolvement + JobLevel + JobSatisfaction + MaritalStatus + 
               MonthlyIncome + NumCompaniesWorked + OverTime + RelationshipSatisfaction + 
               TotalWorkingYears + WorkLifeBalance + YearsInCurrentRole + 
               YearsSinceLastPromotion, family = binomial(link = "logit"), 
             data = train)
summary(model)
######################################################################################
# 测试集训练
test_data <- read.csv("pfm_test.csv", header = TRUE)
names(test_data)[1] <- "Age"
test_data <- test_data[, -c(7,17,22)]
test_data[,ord_var] <- lapply(test_data[,ord_var], function (x) as.factor(x))

test_data$MonthlyIncome <- log(test_data$MonthlyIncome)
# test_data$YearsAtCompany <- log(test_data$YearsAtCompany+1)
# test_data$YearsSinceLastPromotion <- log(test_data$YearsSinceLastPromotion+1)
# test_data$TotalWorkingYears <- log(test_data$TotalWorkingYears+1)
# test_data$NumCompaniesWorked <- log(test_data$NumCompaniesWorked+1)

pred <- predict(model_total.step, test_data, type='response')
result <- ifelse(pred>0.5,1,0)
result <- data.frame(result)
# 保存预测值
k1<-sum(result)/350;k1
write.csv(result,"result_feature_engineering.csv",row.names = F)
# 人工选择
pred <- predict(model, test_data, type='response')
result_by_hand <- ifelse(pred>0.5,1,0)
result_by_hand <- data.frame(result_by_hand)
k2<-sum(result_by_hand)/350;k2
write.csv(result,"result_by_hand.csv",row.names = F)
# 差异
diff <- table(result_by_hand,result, dnn = c("手工值","选择值"));diff
