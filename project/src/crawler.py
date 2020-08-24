import newspaper
from newspaper import Article
import pandas as pd
data = pd.read_csv('../data/uci-news-aggregator.csv',sep=',')
#while(1):
#print(newspaper.popular_urls())
urls = newspaper.popular_urls()
#urls = ['https://www.wsj.com/news/world/middle-east','http://wsj.com','http://nytimes.com','http://www.bbc.co.uk','http://www.npr.org','http://www.reuters.com','http://www.economist.com','http://www.pbs.org','http://bigstory.ap.org','http://cnn.com','http://www.ted.com','http://www.washingtonpost.com','http://www.newyorker.com','http://www.cbs.com']
#print(data.head(5))
url_test = "https://www.nytimes.com/2020/08/08/business/economy/lost-unemployment-benefits.html?action=click&module=Top%20Stories&pgtype=Homepage"
article = Article(url_test)
article.download()
article.parse()
print(article.text)
exit()
for row in data:
    try:
        first_article = Article(url=str(row['URL'], language='en'))
        print(first_article)
        #article = newspaper.build(str(row['URL']), memoize_articles=False, language='en')
        #with open('sample'+str(row['ID']), 'w') as f:
    except:
        print("Error")

    '''print("$$$$ Novi url: ",url)
    cnn_paper = newspaper.build(str(url), memoize_articles=False, language='en')
    print("$$$$ Broj artikala: ",len(cnn_paper.articles))
    j = 0
    for article in cnn_paper.articles:
        try:
            j+=1
            article.download()
            article.parse()
            print(j," : ",article.url)
            if(len(article.text) > 0):
                with open("ex"+str(i), 'w') as f:
                    f.write(article.text)
                    i+=1
        except:
            print("Error")
    '''