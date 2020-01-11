rm(list=ls())
setwd('d:\\Documents\\wuhaoyu\\桌面\\daguan')
library(pROC)

set.seed(1234)
# 数据集准备
data <- read.csv('pfm_train.csv', header=TRUE)
str(data)
data$Attrition<-as.factor(data$Attrition)
data$EmployeeNumber <- as.character(data$EmployeeNumber)
names(data)[1]='Age'
select<-sample(1:nrow(data), length(data$EmployeeNumber), replace=FALSE)
data['MonthlyIncome'] = log(data['MonthlyIncome'])
data['YearsSinceLastPromotion'] = log(data['YearsSinceLastPromotion']+1)
train_df = data[select,]
test_df = data[-select,]
# 不变变量
constant_var <- c("StandardHours", "Over18")
# 高相关变量
high_cor_var <- c("JobLevel","TotalWorkingYears","YearsAtCompany","YearsWithCurrManager", "PerformanceRating","PercentSalaryHike")
# 分类变量
char_var <- c("BusinessTravel", "Department", "EducationField","Gender", "JobRole","MaritalStatus", "OverTime")
# 数据变量W
num_var <- colnames(train_df)[-which(colnames(train_df)%in%c(constant_var,char_var))]
# fitler var
filter_var <- colnames(train_df)[-which(colnames(train_df)%in%c(constant_var,char_var,high_cor_var))]
# 逐步回归
total_model = glm(formula = 
                Attrition ~
                Age+
                BusinessTravel+
                Department+
                DistanceFromHome+
                Education+
                EducationField+
                EnvironmentSatisfaction+
                Gender+
                JobInvolvement+
                JobRole+
                JobSatisfaction+
                MaritalStatus+
                MonthlyIncome+
                NumCompaniesWorked+
                OverTime+
                RelationshipSatisfaction+
                StockOptionLevel+
                TrainingTimesLastYear+
                WorkLifeBalance+
                YearsInCurrentRole+
                YearsSinceLastPromotion,
              data = train_df, 
              family = binomial)
summary(total_model)
total_model_step <- step(total_model,direction = 'both')
summary(total_model_step)
describe(total_model)
model = glm(formula = 
              Attrition ~
              Age + BusinessTravel + DistanceFromHome + EducationField + EnvironmentSatisfaction + Gender + 
              JobInvolvement + JobSatisfaction + MaritalStatus + MonthlyIncome + 
              NumCompaniesWorked + OverTime + RelationshipSatisfaction + 
              WorkLifeBalance + YearsInCurrentRole + YearsSinceLastPromotion,
            data = train_df, 
            family = binomial)
describe(model)
anova(model, test= "Chisq")

fitted.results <- predict(model,test_df,interval="confidence")
fitted.results <- ifelse(fitted.results>0,1,0)
#输出混淆矩阵
table(test_df$Attrition,fitted.results,dnn=c("真实值","预测值"))
accurancy = mean(fitted.results ==test_df$Attrition)
accurancy
xgboost_roc <- roc(test_df$Attrition,as.numeric(fitted.results))
#绘制ROC曲线和AUC值
plot(xgboost_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='模型ROC曲线')

am.data = glm(Attrition ~
                Age+
                BusinessTravel+
                DistanceFromHome+
                EnvironmentSatisfaction+
                Gender+
                JobInvolvement+
                JobRole+
                JobSatisfaction+
                MaritalStatus+
                MonthlyIncome+
                NumCompaniesWorked+
                OverTime+
                RelationshipSatisfaction+
                YearsInCurrentRole+
                YearsSinceLastPromotion,
              train_df, 
              family = binomial)
print(summary(am.data))
anova(am.data, test= "Chisq")

fitted.results <- predict(am.data,test_df,interval="confidence")
fitted.results <- ifelse(fitted.results>0,1,0)
#输出混淆矩阵
table(test_df$Attrition,fitted.results,dnn=c("真实值","预测值"))
accurancy = mean(fitted.results ==test_df$Attrition)
accurancy
rocplot <- roc(test_df$Attrition,as.numeric(fitted.results))
#绘制ROC曲线和AUC值
plot(rocplot, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='xgboost模型ROC曲线')


