# Install
install.packages("tm")  # for text mining
install.package("SnowballC") # for text stemming
install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes
# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")

# Read the text file
filePath <- "C:/Users/kein/Desktop/Kaiying_zhang_assignment3/Part 1/resource/iliad.txt"
text <- readLines(filePath)
# Load the data as a corpus
docs <- Corpus(VectorSource(text))

inspect(docs)

toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")

# Convert the text to lower case
docs <- tm_map(docs, content_transformer(tolower))
# Remove numbers
docs <- tm_map(docs, removeNumbers)
# Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
# Remove your own stop word
# specify your stopwords as a character vector
docs <- tm_map(docs, removeWords, c("blabla1", "blabla2")) 
# Remove punctuations
docs <- tm_map(docs, removePunctuation)
# Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)
# Text stemming
# docs <- tm_map(docs, stemDocument)


dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)

set.seed(142)   
wordcloud(words = d$word, freq = d$freq, min.freq=1)   

set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,max.words=200, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"))


findFreqTerms(dtm, lowfreq = 4)
findAssocs(dtm, terms = "freedom", corlimit = 0.3)
barplot(d[1:10,]$freq, las = 2, names.arg = d[1:10,]$word,
        col ="lightblue", main ="Most frequent words",
        ylab = "Word frequencies")

#do some clusting

# This makes a matrix that is only 15% empty space.
dtms <- removeSparseTerms(dtm, 0.15) 
dtms<-dtms[c(1:2),c(1:20)]
library(cluster)   
# First calculate distance between words
d <- dist(t(dtms), method="euclidian")   
fit <- hclust(d=d, method="ward.D") 
fit

plot.new()
plot(fit, hang=-1)
# "k=" defines the number of clusters you are using   
groups <- cutree(fit, k=5)  

# draw dendogram with red borders around the 5 clusters 
rect.hclust(fit, k=5, border="red") 


### K-means clustering   
library(fpc)   
library(cluster) 
freq <- colSums(as.matrix(dtm))

length(freq)

ord <- order(freq)
wf <- data.frame(word=names(freq), freq=freq)
head(wf)

wfs <- wf[rev(order(wf$freq)),]

top50 <- head(wfs,50)
dtms <- removeSparseTerms(dtm, 0.15) # Prepare the data (max 15% empty space)   
d <- dist(top50, method="euclidian")   
kfit <- kmeans(d, 2)   
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=2, lines=0) 






