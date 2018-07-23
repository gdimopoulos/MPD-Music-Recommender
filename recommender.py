
# coding: utf-8

# In[1]:


import pandas as pd
import sqlite3

conn = sqlite3.connect("musiclibrary.blb")


# In[2]:


##### Load the 'items' table ####

# load the desired columns from the table into a DF
df_items = pd.read_sql_query("select id, title, artist, length, year, album, genre, grouping, path from items;", conn)
# cast path type to str
df_items['path'] = df_items['path'].astype('str')


# In[3]:


##### Load the 'item_attributes' table ####

# load entire 'item_attributes' to a DF
df_attr = pd.read_sql_query("select id, entity_id, key, value from item_attributes;", conn)

# pivot columns to rows
pivot = df_attr.pivot(index='entity_id', columns='key', values='value')

# create column 'id' to merge
pivot['id'] = pivot.index


# In[4]:


# merge the two df on 'id' column
df_item_attr = pd.merge(df_items, pivot, on='id', how='outer')


# In[5]:


excludes = ['id','title','artist','album','length','year','data','soup','path','data_source']
keys = [str(key) for key in df_item_attr.keys().values if key not in excludes]


# In[6]:


def create_soup(x):
    return ' '.join([str(x[key]) for key in keys])
    #return ' '.join([x['genre'], x['grouping'], str(x['year']), str(x['length'])])

df_item_attr['soup'] = df_item_attr.apply(create_soup, axis=1)


# In[7]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df_item_attr['soup'])


# In[8]:


# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[9]:


# Reset index of your main DataFrame and construct reverse mapping as before
df_items = df_items.reset_index()
indices = pd.Series(df_items.index, index=df_items['id'])


# In[10]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(seed, recommendations, cosine_sim=cosine_sim2):

    # Get the index of the movie that matches the title
    idx = indices[seed]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the <length> most similar tracks
    sim_scores = sim_scores[1:recommendations+1]

    # Get the track indices
    track_indices = [i[0] for i in sim_scores]
    
    return track_indices
    #return df_items['title'].iloc[track_indices], df_items['path'].iloc[track_indices]


# In[12]:


seed = 190
recommendations = 20
recommendation_indices = get_recommendations(seed, recommendations, cosine_sim2)

seed_item = df_items[df_items['id']==seed]

print "Getting the top {} most similar tracks for song".format(recommendations)
print "{} by {}\n".format((seed_item['title'].iloc[0]).encode('utf-8'),(seed_item['artist'].iloc[0]).encode('utf-8'))


import mpd

client = mpd.MPDClient()

host = '192.168.1.111'
port = 6600

try:
    client.connect(host, port)
except socket.error:
    print "couldn't connect to mpd server."

c = 0
for index in recommendation_indices:
    c+=1
    print "{}.\t{} - {}".format(c, (df_items['title'].iloc[index]).encode('utf-8'), (df_items['artist'].iloc[index]).encode('utf-8'))
    client.addid(df_items['path'.iloc[index]], c-1)
