# necessary libraries
library(twitteR)
library(tm)
library(e1071)
library(gmodels)
library(ggplot2)
library(wordcloud)

# reading in the csv file
# must set up working directory based on where its located on computer
setwd("D:/RYERSON courses/CKME 136/finalproj")
tweets<-read.csv("socialmedia-disaster-tweets-DFE.csv", header = TRUE, sep = ",")

# removing unnecessary columns
tweets <- tweets[c(6,9,10,11)]

# renaming "choose_one" column to "Relevancy"
names(tweets) <- c("relevancy","keyword", "location","text")
# reordering the columns so it shows "Relevancy" then "text" and all other columns
tweets<-tweets[c("relevancy","text","keyword", "location")]

# adding a level "none" for tweets with no keyword associated with it
levels(tweets$keyword) <- c(levels(tweets$keyword), "none")
noKey<-which(tweets$keyword=="")
for (n in noKey){tweets$keyword[n]<-"none"}


# removing rows where the "relevancy" rating is "Can't Decide"
tweets <- subset(tweets, relevancy == "Relevant" | relevancy == "Not Relevant")

# eliminating the (Can't Decide) from the factor level
tweets$relevancy<- droplevels(tweets$relevancy)

# change tweets to character type
tweets$text<-as.character(tweets$text)

# remove random characters from between keywords
tweets$keyword <- gsub("%20"," ",tweets$keyword)


################# EXPLORATION #################

#checking to see how many NAs in each column
sapply(tweets, function(y) sum(length(which(is.na(y)))))

#checking to see how many 0s are in each column
sapply(tweets, function(y) sum(y==0, na.rm = T))

#checking to see the class of each column
sapply(tweets,class)

# checking the number of tweets and the first few lines stored in tweets
nrow(tweets)
head(tweets)
str(tweets)

# 
table(tweets$relevancy)
pie(table(tweets$relevancy), main = "Proportion of Relevant vs Not Relevant Tweets", col = 2:4)

################# PROCESSING TEXT DATA FOR ANALYSIS #################

#create a corpus
myCorpus <- Corpus(VectorSource(tweets$text))

#Preprocessing the tweets
myCorpus <- tm_map(myCorpus, content_transformer(tolower))

#removing usersnames that appear after the @ symbol
removeUsername <- function(x) gsub("@\\S+", "", x)
myCorpus <- tm_map(myCorpus, content_transformer(removeUsername)) 

# remove urls
removeURL <- function(x) gsub('(f|ht)tp\\S+\\s*',"", x)
myCorpus <- tm_map(myCorpus, content_transformer(removeURL)) 

# based on observing the results, adding this function to catch when two words were separated only by : or ;
sepWords<-function(x) gsub('[:;]', ' ', x)
myCorpus <- tm_map(myCorpus, content_transformer(sepWords)) 

# remove punctuation
myCorpus <- tm_map(myCorpus, removePunctuation)

# remove special characters not caught
removeSpec <-function(x) gsub('[^a-zA-Z0-9]', ' ', x)
myCorpus <- tm_map(myCorpus, content_transformer(removeSpec)) 

# remove numbers
myCorpus <- tm_map(myCorpus, removeNumbers)

# remove stopwords
myStopwords <- c(stopwords('english'))
myCorpus <- tm_map(myCorpus, removeWords, myStopwords)

# strip leading and trailing whitespace
myCorpus <- tm_map(myCorpus, stripWhitespace)

myCorpus <- tm_map(myCorpus, PlainTextDocument)
# to analyze certain documents, this line can be used
# myCorpus[[1]]$content
 for (n in 1:6) {print(dictCorpus[[n]]$content)}
for (n in 1:6) {print(myCorpus[[n]]$content)}

#Stemming
dictCorpus <- myCorpus
myCorpus <- tm_map(myCorpus, stemDocument)

#workaround for stemcompletion method not working in tm package version 0.6

stemCompletion2 <- function(x, dictionary) {
  x <- unlist(strsplit(as.character(x), " "))
  # Unexpectedly, stemCompletion completes an empty string to
  # a word in dictionary. Remove empty string to avoid above issue.
  x <- x[x != ""]
  x <- stemCompletion(x, dictionary=dictionary)
  x <- paste(x, sep="", collapse=" ")
  PlainTextDocument(stripWhitespace(x))
}
myCorpus <- lapply(myCorpus, stemCompletion2, dictionary=dictCorpus)
myCorpus <- Corpus(VectorSource(myCorpus))


# Building a document-term matrix
myDtm <- DocumentTermMatrix(myCorpus, control = list(minWordLength = 1))

################# CREATING TRAINING AND TEST DATASETS #################

# creating the training and test set, randomized
t_train <- sample(nrow(tweets),floor(nrow(tweets)*0.7))
tweet_raw_train <- tweets[t_train,]
tweet_raw_test <- tweets[-t_train,]

# document-term matrix
tweet_dtm_train <- myDtm[t_train,]
tweet_dtm_test <- myDtm[-t_train,]

tweet_corpus_train <- myCorpus[t_train]
tweet_corpus_test <- myCorpus[-t_train]

# check if the subsets are representative of the twitter data
prop.table(table(tweet_raw_train$relevancy))
prop.table(table(tweet_raw_test$relevancy))

################# CREATING INDICATOR FEATURES FOR FREQUENT WORDS #################

#all words included appear in at least 10 messages
tweet_dict <- findFreqTerms(tweet_dtm_train, 10)
tweet_train <- DocumentTermMatrix(tweet_corpus_train, list(dictionary = tweet_dict))
tweet_test <- DocumentTermMatrix(tweet_corpus_test, list(dictionary = tweet_dict))

#converting labels to see if word is included in tweet
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}

# converts each of the columns so that it specifies whether a word is present in the tweet or not
tweet_train <- apply(tweet_train, MARGIN = 2, convert_counts)
tweet_test <- apply(tweet_test, MARGIN = 2, convert_counts)

################# TRAINING A MODEL ON THE DATA (applying Naives Bayes) #################

tweet_classifier <- naiveBayes(tweet_train, tweet_raw_train$relevancy, laplace = 1)

################# EVALUATING MODEL PERFORMANCE #################

# making predictions
tweet_test_pred <- predict(tweet_classifier, tweet_test)

# confusion matrix
CrossTable(tweet_test_pred, tweet_raw_test$relevancy, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted', 'actual'))

######################## DATA DESCRIPTION #######################

tdm <- TermDocumentMatrix(myCorpus, control = list(wordLengths = c(1, Inf)))

#Showing the words that appeared the most in tweets, overall
term.freq <- rowSums(as.matrix(tdm))
# to limit how many words appeared in the bar chart, only words that appeared at least 175 times are included
term.freq <- subset(term.freq, term.freq >=175)
df <- data.frame(term = names(term.freq), freq = term.freq)
df.sort <- df[order(-df$freq),]

# ggplot of words that appeared the most in tweets, overall
bp <- ggplot(df.sort, aes(term, freq))
bp + geom_bar(stat = "identity",aes(fill = freq),na.rm = T) + coord_flip() + xlab("Terms") + ylab("Count") + ggtitle("Words That Appeared in Most Tweets, Overall") 

# which words are associated with keywords used to find tweets?
keywords<-unique(tweets$keyword)
keywords<-as.character(keywords)

# sample of keywords and their associations
val<-sample(length(keywords),4)
few.assocs<-findAssocs(tdm, keywords[val], 0.2)
data <- unlist(few.assocs)

# visualizing the data
dat<-data.frame(words=names(data),perc=as.numeric(data))

ggplot(dat, aes(x=names(data), y=as.numeric(data), fill=names(data))) + geom_bar(colour="black", stat="identity") + guides(fill=FALSE) + xlab("Associated Words") + ylab("value (up to 1) of word association") + ggtitle("Word Associations") + coord_flip()

##################### Creating Word Clouds ####################


# creating a subset of values in cases where tweets were considered "relevant" to a disaster
relevant_tweets <- myCorpus[which(tweets$relevancy=="Relevant")]
# creating a subset of values in cases where tweets were considered "not relevant" to a disaster
irrelevant_tweets <- myCorpus[which(tweets$relevancy=="Not Relevant")]

# converting to PlanTextDocuments 
relevant_tweets <- tm_map(relevant_tweets, PlainTextDocument)
irrelevant_tweets <- tm_map(irrelevant_tweets, PlainTextDocument)

# creating a wordcloud, words must appear at least 125 times to be included
wordcloud(relevant_tweets, min.freq = 125, random.order = FALSE,colors=rainbow(5))
wordcloud(irrelevant_tweets, min.freq = 125, random.order = FALSE,colors=rainbow(3))
