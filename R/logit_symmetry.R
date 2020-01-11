rm(list=ls())
setwd('d:\\Documents\\wuhaoyu\\����\\daguan')

set.seed(1234)
# ����Ԥ����׼��
data <- read.csv('pfm_train.csv', header=TRUE)
names(data)[1]='Age'
data <- data[,-c(23,18)]
str(data)
data$Attrition<-as.factor(data$Attrition)
data$EmployeeNumber <- as.character(data$EmployeeNumber)
describe(data)

data['MonthlyIncome'] = log(data['MonthlyIncome'])
data['YearsSinceLastPromotion'] = log(data['YearsSinceLastPromotion']+1)
describe(data)

# ���ݼ�׼��
select<-sample(1:nrow(data), length(data$EmployeeNumber)*0.7, replace=FALSE)
train_df = data[select,]
test_df = data[-select,]
n_dataset = train_df[which(train_df$Attrition==0),]
p_set = data[which(data$Attrition==1),]
p_select <- sample(1:nrow(p_set), length(p_set$EmployeeNumber)*2, replace=TRUE)
p_dataset <- data[p_select,]
train_df = rbind(n_dataset,p_dataset)

# ȫģ��
total_model = glm(formula = 
                    Attrition~
                    Age+
                    BusinessTravel+
                    Department+
                    DistanceFromHome+
                    Education+
                    EducationField+
                    EnvironmentSatisfaction+
                    Gender+
                    JobInvolvement+
                    JobLevel+
                    JobRole+
                    JobSatisfaction+
                    MaritalStatus+
                    MonthlyIncome+
                    NumCompaniesWorked+
                    OverTime+
                    PercentSalaryHike+
                    PerformanceRating+
                    RelationshipSatisfaction+
                    StockOptionLevel+
                    TotalWorkingYears+
                    TrainingTimesLastYear+
                    WorkLifeBalance+
                    YearsAtCompany+
                    YearsInCurrentRole+
                    YearsSinceLastPromotion+
                    YearsWithCurrManager,
                  data = train_df, 
                  family=binomial(link="logit"))
summary(total_model)
total_model_step <- step(total_model,direction = 'both')
summary(total_model_step)
# AIC�x��ģ��
aic_model = glm(formula = 
                  Attrition~
                  BusinessTravel+
                  DistanceFromHome+
                  EnvironmentSatisfaction+
                  Gender+
                  JobInvolvement+
                  JobSatisfaction+
                  MaritalStatus+
                  MonthlyIncome+
                  NumCompaniesWorked+log
                  OverTime+
                  PercentSalaryHike+
                  StockOptionLevel+
                  YearsInCurrentRole+
                  YearsSinceLastPromotion+
                  YearsWithCurrManager,
                data = train_df, 
                family=binomial(link="logit"))
summary(aic_model)
fitted.results <- predict(aic_model,test_df,interval="confidence")
fitted.results <- ifelse(fitted.results>0,1,0)
#�����������
table(test_df$Attrition,fitted.results,dnn=c("��ʵֵ","Ԥ��ֵ"))
accurancy = mean(fitted.results ==test_df$Attrition)
accurancy
rocplot <- roc(test_df$Attrition,as.numeric(fitted.results))
#����ROC���ߺ�AUCֵ
plot(rocplot, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='logitģ��ROC����')

# �˹��x��
am.data = glm(formula = 
                    Attrition~
                    BusinessTravel+
                    DistanceFromHome+
                    EnvironmentSatisfaction+
                    Gender+
                    JobInvolvement+
                    JobSatisfaction+
                    MaritalStatus+
                    MonthlyIncome+
                    NumCompaniesWorked+
                    OverTime+
                    RelationshipSatisfaction+
                    StockOptionLevel+
                    YearsInCurrentRole+
                    YearsSinceLastPromotion,
                  data = train_df, 
                  family=binomial)
summary(am.data)

fitted.results <- predict(am.data,test_df,interval="confidence")
fitted.results <- ifelse(fitted.results>0,1,0)
#�����������
table(test_df$Attrition,fitted.results,dnn=c("��ʵֵ","Ԥ��ֵ"))
accurancy
xgboost_roc <- roc(test_df$Attrition,as.numeric(fitted.results))
accurancy = mean(fitted.results ==test_df$Attrition)

#����ROC���ߺ�AUCֵ
plot(xgboost_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='xgboostģ��ROC����')


