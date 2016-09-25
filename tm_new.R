setwd("E:/Mstat Courses/txt min")

# required libraries

library(readr)
library(tm)
library(wordcloud)
library(png)

# reading the data from the downloaded csv
cat("Read data ...\n")
reviews <- read.csv("Reviews.csv")
library("ff")
reviews <- read.csv.ffdf(file = "Reviews.csv", header = TRUE, VERBOSE = TRUE, first.rows = 10000, 
                         next.rows = 50000, colClasses = NA)
# initial exploration
summary(reviews)
summ <- reviews$Summary

# creating a cropus
docs <- Corpus(VectorSource(summ))

# converting variables to lower case
docs <- tm_map(docs, content_transformer(tolower))

docs <- Corpus(VectorSource(summ))

# removing patterns
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")
docs <- tm_map(docs, tolower)

# removing other punctuation
docs <- tm_map(docs, removePunctuation)

# removing white space
docs <- tm_map(docs, stripWhitespace)

# removing any numbers present
docs <- tm_map(docs, removeNumbers)

# remving the common stop words
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, PlainTextDocument)

# normalizing
docs <- tm_map(docs, stemDocument, "english")

# creating DTM
dtm <- DocumentTermMatrix(docs)

# chossing terms sparser then 99.5%
dtm1 <- removeSparseTerms(dtm, 0.995)
m <- as.matrix(dtm1)
dim(m)

# checking words that occur more than 20 times
findFreqTerms(dtm1, lowfreq = 20)

# initial exploration
(kcl <- kmeans(dtm1, 2, nstart = 25))
kcl$size
clust <- factor(kcl$cluster)
summary(clust)

# check if the normalization works
dtm1 <- removeSparseTerms(dtm, 0.997)
rowTotals <- apply(dtm1, 1, sum)  #Find the sum of words in each Document
input <- dtm1/rowTotals

# run kmeans for all clusters up to 25
for (i in 1:25) {
  # Run kmeans for each level of i, allowing up to 100 iterations for convergence
  kmeans <- kmeans(x = dtm1, centers = i, iter.max = 100)
  
  # to be ysed for plotting
  cost_df <- rbind(cost_df, cbind(i, kmeans$tot.withinss))
  
}

# cluster chosen
(kcl <- kmeans(dtm1, 5, nstart = 25))