library(tidyverse)


raw_data <- read.csv("C:/Users/Mexbol/Desktop/maestria/Computational_Economics/FINAL/All estimation raw data.csv")
raw_comp_set_data <- read.csv("C:/Users/Mexbol/Desktop/maestria/Computational_Economics/FINAL/raw-comp-set-data-Track-2.csv")
raw_data <- raw_data[, c(3:32)]
raw_data <- rbind(raw_data, raw_comp_set_data)

test <- aggregate(RT~GameID+block, data=raw_data, FUN="mean")
test <- rename(test, meanRT=RT)
raw_data <- merge(raw_data, test, by=intersect(names(raw_data), names(test)), all.x=T)

raw_data$Moretime <- ifelse(raw_data$RT>raw_data$meanRT, 1, 0)

test <- aggregate(B~GameID+block, data=raw_data, FUN="mean")
test <- rename(test, meanB=B)

test2 <- aggregate(B~GameID+block + SubjID, data=raw_data, FUN="mean")
test2 <- rename(test2, meanB_subject=B)

raw_data <- merge(raw_data, test, by=intersect(names(raw_data), names(test)), all.x=T)
raw_data <- merge(raw_data, test2, by=intersect(names(raw_data), names(test2)), all.x=T)

raw_data$B_more <- ifelse(raw_data$meanB_subject >raw_data$meanB, 1, 0)

raw_data$consistent <- ifelse((raw_data$meanB_subject-raw_data$meanB)/raw_data$meanB<=0.25 |
                                (raw_data$meanB_subject-raw_data$meanB)/raw_data$meanB>=-0.25 , 1, 0)
raw_data$consistent <- ifelse(raw_data$meanB_subject==0 & raw_data$meanB==0, 1, 0)
raw_data$RT <- ifelse(is.na(raw_data$RT), median(raw_data$RT, na.rm = T), raw_data$RT)
raw_data$meanRT <- ifelse(is.na(raw_data$meanRT), median(raw_data$meanRT, na.rm = T), raw_data$meanRT)
raw_data$Moretime <- ifelse(is.na(raw_data$Moretime), median(raw_data$Moretime, na.rm = T), raw_data$Moretime)

raw_data <- raw_data[, c(27, 31:36)]
write.csv(raw_data, file = "C:/Users/Mexbol/Desktop/maestria/Computational_Economics/FINAL/attention_var.csv")

cor_data <- read.csv("C:/Users/Mexbol/Desktop/maestria/Computational_Economics/FINAL/corr_data.csv")
mydata.cor = cor(cor_data)

