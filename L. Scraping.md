# Scraping

Example of scraping using Requests and BeautifulSoup librairies. Scrap list of events in Paris in 2021.
<br> Theory introduced by: https://www.dataquest.io/blog/web-scraping-python-using-beautiful-soup/

Notions: *HTML, tag, body, parent, child, division, class, id*
<br>
<br>
### Importation
First import the requests library, and then download the page using the requests.get method:
```
import requests
page = requests.get("https://www.cityzeum.com/evenement/paris")
```
After running our request, we get a Response object. 
<br>This object has a status_code property, which indicates if the page was downloaded successfully:
```
page.status_code
```
### Parsing
Parsing a page with BeautifulSoup. Parsing = convert (découper/disséquer). Here, convert html to exploitable format
<br>We can use the BeautifulSoup library to parse this document, and extract the text from the p tag.
```
from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content, 'html.parser')
```

<br>We can now print out the HTML content of the page, formatted nicely, using the prettify method on the BeautifulSoup object.
```
print(soup.prettify())
```

<br>As all the tags are nested, we can move through the structure one level at a time. We can first select all the elements at the top level of the page using the children property of soup.
```
list(soup.children)
# Let’s see what the type of each element in the list is:
[type(item) for item in list(soup.children)]
```
As we can see, all of the items are BeautifulSoup objects:

<br>We can now select the html tag and its children by taking the third item in the list:
```
html = list(soup.children)[2]
```
<br> Now, we can find the children inside the html tag:
```
list(html.children)
# As we can see above, there are two tags here, head, and body. We want to extract the text inside the p tag, so we’ll dive into the body:
body = list(html.children)[3] # The 'n' counts as 1
# Now, we can get the p tag by finding the children of the body tag:
list(body.children)
# We can now isolate the p tag:
p = list(body.children)[1]
# Once we’ve isolated the tag, we can use the get_text method to extract all of the text inside the tag:
p.get_text()
```



### Finding all instances of a tag at once
<br> If we want to extract a single tag, we can instead use the find_all method, which will find all the instances of a tag on a page.
```
soup.find_all('p')
```

### Let's go back to the example
<br>Now, find the division that contains the information you want. You can search by class, tag or id. Id is unique per page so, if it exists, easier
```
# Prepare the dataframe
titles=[] 
descriptions=[] 

list_events = soup.find(id='list_events') # from the object "soup", find the division with the id "list_events"
events = list_events.find_all(class_='description') # fin all the "description" classes --> inspect the source code on the webpage and find the division and the architecture
for a in events: # for each event/child
    try:
        title=a.find('a').get_text() # find tag "a" which refers to the tile
        description = a.find('p').get_text() # find tag "p" which refers to the description
        titles.append(title)
        descriptions.append(description)
    except:
        print(title)
```
<br>Store the data in a dataframe and save it as .csv
```
df_events = pd.DataFrame({'title': titles, 'description': descriptions})
df_events.to_csv('events_list.csv', index=False)
```


