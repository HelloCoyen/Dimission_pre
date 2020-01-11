rm(list=ls())
library(caret)
library(cluster)
setwd("C:\\Users\\wuhaoyu\\Desktop\\DG")
# 数据加载，基础预处理
train_Xy <- read.csv("pfm_train.csv", header = TRUE, stringsAsFactors = FALSE)
train_X <- train_Xy[-grep("Attrition", colnames(train_Xy))]
train_y <- train_Xy["Attrition"]
train_y$Attrition <- as.factor(train_y$Attrition)
test_X <- read.csv("pfm_test.csv", header = TRUE, stringsAsFactors = FALSE)

# 数据清洗，列名
names(train_X)[1] <- "Age"

# 删除无效变量
train_X <- train_X[-grep("Over18|StandardHours|EmployeeNumber", colnames(train_X))]

# 组合变量
train_X <- unite(train_X, "job", Department, JobRole , remove = TRUE)

select_var <- c('Age',
                'BusinessTravel',
                'job',
                'DistanceFromHome',
                'EducationField',
                'EnvironmentSatisfaction',
                'Gender',
                'JobInvolvement',
                'JobLevel',
                'JobSatisfaction',
                'MaritalStatus',
                'MonthlyIncome',
                'NumCompaniesWorked',
                'OverTime',
                'RelationshipSatisfaction',
                'TotalWorkingYears',
                'WorkLifeBalance',
                'YearsAtCompany',
                'YearsInCurrentRole',
                'YearsSinceLastPromotion',
                'YearsWithCurrManager'
)
names(train_X)
train_X<-train_X[,select_var]
# 全变量
all_columns = names(train_X)

# 数值变量
num_var <- c("Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked", 
             "PercentSalaryHike", "StockOptionLevel", "TotalWorkingYears", "YearsAtCompany",
             "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager")

# 非数值变量
unnum_var <- all_columns[-which(all_columns%in%num_var)]

# # 特征类型转换
dummies <- dummyVars(~BusinessTravel+job+Education+EducationField+
                      EnvironmentSatisfaction+Gender+JobInvolvement+
                      JobLevel+JobSatisfaction+MaritalStatus+OverTime+
                      PerformanceRating+RelationshipSatisfaction+TrainingTimesLastYear+
                      WorkLifeBalance, data = train_X)
dummies <- dummyVars(~BusinessTravel+job+EducationField+
                      EnvironmentSatisfaction+Gender+JobInvolvement+
                      JobLevel+JobSatisfaction+MaritalStatus+OverTime+
                      RelationshipSatisfaction+
                      WorkLifeBalance, data = train_X)
df_all_ohe <- as.data.frame(predict(dummies, newdata = train_X))
df_all_combined <- cbind(train_X[,c(which(colnames(train_X) %in% num_var))], df_all_ohe)
df_all_combined <- lapply(df_all_combined, function(x) scale(x, center = T, scale = T))
df_all_combined <- as.data.frame(df_all_combined)
str(df_all_combined)
# 建数据集
cl <- kmeans(df_all_combined,2); cl
centers <- cl$centers[cl$cluster,]
distances <- sqrt(rowSums((df_all_combined-centers)^2))
outerliens <- order(distances,decreasing = T)[1:9]
ot <- train_Xy[outerliens,]
ot1 <- ot[which(ot$Attrition==0),]
View(ot)

print(outerliens)
plot(df_all_combined[,c("DistanceFromHome","MonthlyIncome")],pch="o",
     col=cl$cluster,cex=0.3)
points(cl$centers[,c("DistanceFromHome","MonthlyIncome")],col=1:3,
       pch=8,cex=1.5)
points(df_all_combined[outerliens,c("DistanceFromHome","MonthlyIncome")],
       col=4,
       pch="+",cex=1.5)

sum(cl$cluster-1)
result = cl$cluster-1
write.csv(result, "K-means_result.csv")
cl$ifault

cl2 <-pam(train_X,2)
cl2$clustering-1
sum(cl2$clustering-1)
result2 = cl2$clustering-1
write.csv(result2, "K-means_resul2t.csv")
cl$ifault
plot(df_all_combined[,1:8], col = cl$cluster, main="Kmeans Cluster")
2> points(cl$centers, col = 1:3, pch = 10, cex = 4)