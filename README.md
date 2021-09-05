# The International Relations of Twitter

## By Viktor Avdulov, Sean Cafferty, and Asadur Chowdury
## University of Michigan - Capstone Project - 2021

### Introduction

Thucydides’s ‘History of the Peloponnesian War’ is generally considered the first scholarly work that attempts to make sense of the interactions between societies. His exploration of the dynamics of the bloody conflict between Athens and Sparta established the concept that such interactions both could and should be studied. Nearly two millennia later, Machiavelli was attempting to disentangle the complexity of the Holy Roman Empire’s campaigns against Italian city states, while Hobbes was seeking to understand the English Civil War. As seminal as these works are, these early explorations of power and intersocietal interaction noticeably lack the political entities that we take for granted today––the nation state. It was not until the Treaty of Westphalia of 1648 that the concept of nationhood became established as a spatially bound monopoly on violence. Over time, various forms of nation statehood spread throughout the world, either as a result of or response to Europe’s expanding empires. Meanwhile, nationhood had become an ontological entity––something worthy of analysis and measurement that was readily incorporated into Enlightenment-era notions of scientific rationality––think Adam Smith’s ‘The Wealth of Nations’. Eventually, the study of international relations evolved into a more recognizable discipline, and the unfathomably brutal wars of the early 20th century galvanized various ‘Great Debates’ about the ‘true’ nature of the international order. As nuclear weapons raised the stakes for shifting borders and nationhood became the only viable vehicle for the emancipation of colonized peoples, a world of largely fixed nation states emerged. But how should we study this internationalness? As a discipline, international relations largely lacks ontological and epistemological consensus. Is ‘the international’ the result of materiality (e.g. ‘who has the most nukes?’), or is it made of words? Can ‘the international’ even be studied? Ultimately, these various interpretations of the international order (or lack thereof) have shaped the world we live in and the ways in which we understand ourselves. Even so, international relations is a discipline mired in theory, while the actual day-to-day of international interactions is maintained by practitioners––business people, embassy staff, military personnel, and numerous other stakeholders. As such, our data science team sought to utilize the power of data engineering to create a tool that allows such stakeholders to better understand something as complex and evolving as ‘the international’. 

In an effort to make sense of ‘the international’, we chose to study Twitter accounts of embassies. These Twitter embassy accounts are a small, but telling, portion of the narrative world-building that sustains ‘the international’. Compared to the news outlets, print media, and differing forms of online engagement, Twitter represents but a miniscule slice of the information flows that emanate from various foreign ministries. Notwithstanding, Twitter is a useful media artifact for studying international relations insofar as these means of communication represent a distillation of a nation’s self-image as well as its vision of/for the wider world. Embassy Twitter data is thus a rich repository of ideas, practices, symbols, motivations, and intentions that interact to sustain ‘the international’. But how can we harness the power of this data in order to gain actionable insights into the workings of the world? 

Our data science team proposes a heuristic approach driven by data mining and NLP techniques in order to shape the content of these millions of tweets into a navigable framework. In doing so, we hope to create a tool that enables scholars and policymakers to discover mathematical patterns that reflect qualitative insights. We thus propose the International Relations of Twitter dashboard as a tool for dissecting the interactions of nation states on Twitter. In order to demonstrate the potential of this approach, we have created a prototype of the dashboard application that includes the diplomatic Twitter footprint of the United States and Russia, totalling nearly 300 active accounts. Though the Cold War rivals allow for ample exploration of the world of diplomatic messaging, we hope that future iterations of the dashboard application might include the embassy networks of all nation states. Ultimately, we think that this application would not only be a tool to answer questions, but also a framework for finding the right questions to ask. As we demonstrate below, our dashboard app is a tool capable of providing insights relevant to scholars and professionals working in a wide variety of domains. 

![Alt text](/assets/screenshot_dash.png?raw=true "Optional Title")

## Dashboard Features:
1. Explore - Network Level - 
   A tool for exploring the diplomatic messaging across ALL the embassies of a particular country.
2. Explore - Embassy Level - 
   A tool for exploring the diplomatic messaging within a particular embassy.
3. Explore - Comparative Time Series - 
   A tool for finding similarities and differences in the behavior of embassy accounts.
4. Hashtag Dynamics Networks - 
   A tool for understanding the ways in which hastags are used across networks.
5. Topic Modeling Interface - 
   A tool for discovering unseen topics in the text data of each embassy.
6. Network Analysis Interface - 
   A tool for understanding the co-occurrence of named entities in the text data.
7. Document Similarity Dashboard - 
   A tool for understanding semantic patterns within countries.
8. Word Embeddings Dashboard - 
   A tool for understanding semantic patterns about countries. 

## Launching the app
1. First install all necessary package using either pip/pip3:
   
   ```shell
   pip install -r requirements.txt
   ```
2. Next, launch the app by running index.py
  
  ```shell
  python3 -m index.py
  ```

### Full Embassy Dashboard Data
The version available in the Git repository only contains a sample of this larger dataset, which consists of all of the embassies. See this assets folder for complete files
https://drive.google.com/drive/folders/1M5ODnjcNyYR4Xt5OJKE7Z1iuqRsgMaoB?usp=sharing
