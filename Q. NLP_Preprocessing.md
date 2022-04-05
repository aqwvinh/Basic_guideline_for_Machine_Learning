# Theory and Code for NLP

One thing to remember: **better use ```Spacy``` library because every token of a spacy model has multiple attributes**

### Tokenization
Tokenization is the process of **extracting tokens from a text file/document**. These tokens (considered as text units) will serve as **variables for modelization**
<br>You can use the library ```NLTK```, which gives you a list of elements

```
from nltk.tokenize import sent_tokenize, word_tokenize
# Tokenize sentences
df['sentences_new'] = df['sentences_new'].apply(lambda s: sent_tokenize(s))
# Tokenize words
df['text_new'] = df['text_new'].apply(lambda x: word_tokenize(x))
```

### Stop words
Need to remove stop words because they don't bring any valuable information
<br>You can use the libraries ```Spacy``` or ```NLTK```again

Using Spacy
```
# Importing spaCy and creating nlp object
import spacy
nlp = spacy.load('en_core_web_sm')
stopwords = spacy.lang.en.stop_words.STOP_WORDS
list_stopwords=list(stopwords)

# Preprocessing the text
text=nlp(text)
text=[token for token in text if not token.is_stop and not token.is_punct]
```

Using NLTK
```
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
# remove stopwords from perso and nltk stop_words_list
nltk_stop = stopwords.words('english')
file = open("perso_stopwords.txt")
perso_stop = file.read().split("\n")
stop_words = nltk_stop + perso_stop

df['text_clean'] = df['text_new'].apply(lambda x: [word for word in x if word not in stop_words ])
```

### Stemming and Lemmatization
Stemming = reduce a word to its "root" form
<br>Lemmatization = same but the root is correct and meaningful (like infinitif for a verb)
<br>This steps aims at reducing by homogenizing words

#### Lemmatization (Use Spacy)
Using NLTK (most common use)
```
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# We use POS tag of words to improve lemmatizer's results
# We use nltk.pos_tag and a mapping dict to get the right POS
from nltk.corpus import wordnet
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
    
for word in token_list:
    print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
```

Using Spacy
```
def lemmatize(df):
    nlp = en_core_news_md.load()
    col = []
    for i in range(df.shape[0]):
        row = []
        for j in range(len(df['text_clean'][i])):
            word = nlp(df['text_clean'][i][j])[0].lemma_
            row.append(word)
        col.append(row)
    df['text_lemma'] = col
    return df
df = lemmatize(df)
```

#### Stemming
```
from nltk.stem import PorterStemmer
ps = PorterStemmer()
for word in token_list:
    print(ps.stem(word))
```

### POS tagging (Part Of Speech Tagging) (Use Spacy)
Important step because it gets the type of the word (verb, adverb, adjective, noun etc)
<br>It helps the model to better perform

Using NLTK
```
#take a list as an input and return you the same list except that now you have a tuple (word, POS)
token_list = [word for word in word_tokenize(text)] 
pos_list = nltk.pos_tag(token_list)
print(pos_list)
```

Using Spacy
<br>In spaCy, the POS tags are present in the attribute of ```Token``` object. 
<br<You can access the POS tag of particular ```token``` through the ```token.pos_``` attribute.
```
# Create a function to remove tokens of a certain category

# Storing the junk POS tags  
junk_pos=['X','SCONJ'] # "X" POS are like "etc", "i.e." and "SCONJ" are subordinating conjunction, e.g. if, while, that
# You can find the POS list here: https://stackoverflow.com/questions/40288323/what-do-spacys-part-of-speech-and-dependency-tags-mean

# Function to check if the token falls in the JUNK POS category.
def remove_pos(word):
    flag=False
    if word.pos_ in junk_pos:
        flag=True
    return flag

# Creating a new doc without the un-required tokens
revised_text=[token for token in text if remove_pos(token)==False]

# Printing POS tags present in the new document.                   
all_tags = {token.pos: token.pos_ for token in revised_text}
print(all_tags)
```

### Named entity recognition (use Spacy)
Detect a pre-defined category for some words ==> find companies, personnalities etc.
<br>It is a very useful method especially in the field of clasification problems and search engine optimization.
<br> Steps for NER:
- Tokenize
- POS tagging

Using NLTK (for small texts)
```
from nltk import word_tokenize, pos_tag
tokens=word_tokenize(sentence)
tokens_pos=pos_tag(tokens)
print(tokens_pos)

#Next, to obtain the named entities of the text , ```nltk``` provides ne_chunk method.

# Importing ne_chunk from nltk and download requirements
from nltk import ne_chunk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Print all the named entities
named_entity=ne_chunk(tokens_pos)
print(named_entity)
```

Using Spacy (better for huge data)
<br>Every token of a spacy model, has an attribute ```token.label_``` which stores the category/ label of each entity.
<br>The few common labels are:
- ‘ORG’ : companies,organizations,etc..
- ‘PERSON’ : names of people
- ‘GPE’ : countries,states,etc..
- ‘PRODUCT’ : vehicles,food products and so on
- ‘LANGUAGE’ : Names of different languages

```
# Find list of names from an article

news_articles='Restaurant inspections: Space Coast establishments fare well with no closures or vermin Despite strong talk Tallahassee has a lot more to do for Indian River Lagoon | Opinion Kelly Slater-coached surfer from Cocoa Beach wins national title in Texas Despite parachute mishap Boeing Starliner pad abort test a success Vote for FLORIDA TODAY Community Credit Union Athlete of the Week for Oct 28-Nov 2 Brevard diners rave about experiences at Flavor on the Space Coast restaurants Heres what you need to know about Tuesdays municipal elections in Brevard County Save the turtles rally planned to oppose Satellite Beach hotel-condominium project Elon Musk wants humans to colonize space Brad Pitt shows thats easier said than done Google honors actor Will Rogers with new Doodle SpaceX targeting faster deployment of Starlink internet-beaming satellites Rockledge Cocoa hold while Merritt Island gains traction in state football polls Boeing launches and lands Starliner capsule for pad abort test in New Mexico Eight of the 10 most dangerous metro areas for pedestrians are in Florida report says Brevard faith leaders demand apology in open letter to Bill Mick How a Delaware researcher discovered a songbird can predict hurricane seasons Open doors change lives: United Way of Brevard kicks off 2019 campaign Florida man pleads guilty to killing sawfish by removing extended nose with power saw One-woman comedy Excuse My French at Viera Studio dives into authors mind Letters and feedback: Sept 19 2019 Viera talk will highlight technology used for dementia patients Adamson Road open after dump truck crash spilled sand and fuel one injured From the mob to mob movies with Robert De Niro: Meet Larry Mazza former hitman Irishman consultant Tropical Storm Jerry getting stronger in Atlantic; Hurricane Humberto nearing Bermuda Palm Bay Halloween party homicide suspect turns self in police say Fall playoffs: Satellite XC boys win region; Titusville swimmers second Sounds of celebration during Puerto Rican Day Parade in Palm Bay Tractor trailer fire extinguished on I-95 in Cocoa lanes open Logan Harvey bowls a 279 to help Astronaut beat Space Coast in battle of undefeated teams Letters and feedback: Sept 18 2019 Accelerated bargaining a bust as teachers union walks out on second day of pay talks Traffic Alert: Puerto Rican Day Parade set for Palm Bay County commissioners debate biosolids at first of two public hearings US 1 widening in Rockledge under study along with sidewalks and bike lanes Letters and feedback: Nov 3 2019 Titusville man indicted on first-degree murder charge Rockledge Cocoa lead 8 Brevard HS football teams into playoffs Space Coast Junior Achievement accepting business hall of fame nominations Why Brevards proposed moratorium on sewage sludge isnt enough |Rangel Brevard high school football standings after Week 4 Starbucks might build new store at old Checkers site in West Melbourne on US 192 Police search for Palm Bay shooting suspect in Orlando area Does your drivers license have a gold star? Well you need one Cocoa man charged with arson in blaze at triplex where 16 children were celebrating a birthday Unit and lot owners may inspect their ledgers upon written request Palm Bay veteran officer Nelson Moya chosen as police chief By the numbers: Your link to every state salary and what high ranking Florida officials make Longtime state county legislative aide Furru honored by County Commission Week 11: Rockledge pulls out BBQ win; EG Palm Bay Astro win rivalry games County commissioners allocate $446 million to repair South Beaches after Hurricane Dorian Tour of Brevard: Holy Trinity Tigers County Commission begins shakeup of its bloated advisory boards while adding term limits Mystery Jesus plaque washes ashore near Melbourne Beach thanks to Hurricane Humberto waves Widow of combat veteran who died after fight at Brevard Jail appeals to governor Health calendar: June 27-July 4 Going surfing this weekend? Check our wave forecast Have you ever seen a turtle release? Heres your chance Darkness turns to day as SpaceX Falcon Heavy launches from Kennedy Space Center Losing hair in strange places? It could be Alopecia areata Golf tip: Pairing and tee times on the PGA Tour NOAA updates projections saying above normal season could see 5-9 hurricanes 2-4 of them major Cape Canaveral Hospital relocation: Loss of a landmark hopeful future for healthcare and land How do I get adult child to move out start own life? The Atlas V rockets jellyfish in the sky Scores: High school football Week 11 in Brevard County Florida gator caught on camera munching on plastic at St Marks National Wildlife Refuge Q&A: New Melbourne head football coach Tony Rowell Does the First Amendment go too far? | Opinion FHSAA says it has a contingency plan if football associations strike but wont reveal it Tropical Depression Ten forms in central Atlantic; Hurricane Humberto continues to strengthen Updates: Watch SpaceX launch its Falcon Heavy rocket from Kennedy Space Center in Florida Letters and feedback: May 9 2019 The Good Stuff: Soccer star makes Wheaties cover cookies in space and more Highly anticipated pad abort test of Boeingâ€™s Starliner spacecraft is Monday morning Health calendar: Sept 19-26 Naked man running through Titusville Walmart parking lot arrested Lacking money and manpower West Melbourne considers swapping school SROs for armed guards Worst best airports in country: 3 Florida airports make Top 5 worst list Colorado STEM school shooting impacted my family: When will we say enough? | Bellaby Human ashes Legos and a sandwich: Strange things we sent into space Massive 2600-acre burn underway near Brevard County line Health Pro: Trip to Philippines validated nurses career decision Harry Anderson Palm Bays Sitting Man exposes gaps in helping the homeless Brevard School Board splits 4-1 in favor of district plan for teacher pay; union vows war Whos No 1? See the Brevard high school football rankings after Week 4 3 people taken into custody after multiple agencies tracked them at high speeds across Central Florida As dawn breaks over Florida ULA Atlas V rocket lights up morning sky Why breast cancer was called Nuns disease Space Coast-based Rocket Crafters partners with Swiss tech giant RUAG Investigation continues for Palm Bay double killing in 2018 Canâ€™t breastfeed? Tips to provide same nutrition to baby Former Florida Tech golfer Iacobelli wins playoff in Michigan for third pro title Baseball bliss every 50 years: Letâ€™s go Mets! Letâ€™s go Nats! 5 tips to make your holiday travel stress-free Updates: ULA launches Atlas V rocket military satellite from Cape Canaveral Andrew Reed bowls 274 in Eau Gallie win over Satellite Space industry propels Brevard economy into optimistic future; highlights from EDC investors meeting Governor DeSantis signs bill geared toward expanding vocational training Outage leaves 12500 customers without power in Melbourne for nearly an hour Weak cold front brings fall weather after unusually warm October on Space Coast 321 Launch: The space news you might have missed Teacher pay: Union opens pay talks with raise demand supplement for veteran teachers Tour of Brevard: Bayside Bears Youth fishing tournament coming Saturday to Melbourne President Trump heres what Hurricane Michael damage still looks like today Letters and feedback: Aug 8 2019 Man leads police on 100 mph chase through Melbourne Palm Bay Puerto Rican parade to take place this weekend in Palm Bay Viera-based USSSA Pride depart from pro softball league Tip of organized theft ring targeting boats motors leads to arrest of 3 men in Melbourne Multiple crashes slow I-95 traffic near Melbourne Is it time to reconsider how free speech should be? | Bill Cotterell SPCA of Brevard takes influx of cats and dogs from Glades County Brevard residents share some of their favorite recent dining experiences Tuesdays brewery running tour stops at Florida Beer Co in Cape Canaveral 89-year-old Tallahassee Grandma Bunny battles 6-foot snake after it eats visiting birds The horrors of Dorian parallel another nearby tragedy of historic proportions EFSCs Carter Stewart named NJCAA Region 8 Pitcher of the Week Titusville teenager charged with murder of grandfather If Trump bans vape flavors Brevard shops say theyll be out of business Cabana: NASAs Artemis is space explorations next giant leap | Opinion Florida continues cutting phone cords BSOs Confessore starting 25th year as music director High school sports results: May 7 Libraries busy with summer reading programs Four hurricanes in six weeks? Remember 2004 the year of hurricanes Letters and feedback: Nov 2 2019 Letters and feedback: May 8 2019 As Brevard kids head back to school here are five things to make lunchtime happy healthy Palm Bay police investigate overnight shooting that left one dead Rent takes the stage at two playhouses Cinderella Brighton Beach Memoirs arrive on Brevard stages this weekend Junes Saharan dust setting up right conditions for red tide but still too soon to tell Wizarding World of Harry Potter: Dark Arts unleashed on Hogwarts Castle at Universal Orlando Health First restructuring: What we know (and what we dont) From Jack in the Box to In-N-Out Burger: 7 restaurant chains missing in Florida SpaceX anomaly: No further action needed on Crew Dragon explosion cleanup Vietnam War mural pits residents vs Florida community Matter settled  unhappily British cruise line Marella to sail from Port Canaveral in 2021 Kids are at risk as religious exemptions to vaccines spike in Florida Brevard County crime rate falls in 2018 reflecting statewide trend according to FDLE report Florida-Georgia FSU-Miami: Why Saturday afternoon is a college football party in Florida Restaurants in Viera Suntree Port Canaveral Palm Bay Melbourne Beach draw Flavor raves Vote for the FLORIDA TODAY Community Credit Union Athlete of the Week for Sept 9-14 Whats happening May 15-21: Concert in Cocoa summer camps fun fair in Melbourne Port Authority Secretary/Treasurer Harvey seeks more public input at evening meetings SpaceX Falcon Heavy rocket launch: Where to watch from Space Coast Treasure Coast Volusia County Tallahassee Here are 10 of Arizonas most photogenic spots Some of them might surprise you Rockledge police search for suspects after armed robbery Kardashian production features Palm Bay homicide involving sisters-in-law FLORIDA TODAY journalist The holidays are almost here; Space Coast cooks are prepping special meals now How to watch tonights SpaceX Falcon Heavy launch from Kennedy Space Center in Florida New state budget spending wish list includes bug studies lionfish contests stronger coastlines Wheaties: Team USA soccer star Ashlyn Harris make cereal cover and it costs $23 a box Letter carriers Stamp Out Hunger food drive is Saturday May 11 Suzy Fleming Leonard: Fourth-graders talk about journalism share favorite restaurants Falcon Heavys night launch may be seen across portion of Sunshine State Police: Palm Bay teacher jerked autistic student from bus faces abuse charge Kicking axe: Urban axe throwing makes its debut in Brevard as trending sport stress reliever Palm Bay official Andy Anderson hired as Mexico Beach city administrator on Florida Panhandle Daddy Duty: Are we celebrating Mothers Day the right way? Father of 6 dies trying to save sons from rip current in Cocoa Beach Its time we address lack of affordable housing on Space Coast | Opinion Orlando Pride soccer player from Brevard diagnosed with breast cancer Space Coast bus system: Lets peck away at this problem County Commission chair says Do lovebugs have a purpose beyond annoying us all twice a year? Hurricane Humberto continues to strengthen swells expected to affect East-Central Florida surf One more launch this week; Things to know about ULA Atlas V launch from Cape Canaveral Free self-defense class for women this Friday in Cocoa Beach The farce behind blaming video games for El Paso Dayton mass shootings | Rangel Teacher pay talks take new shape as school district and union vow to work together First-place USSSA Pride on hot streak; Bennett on ESPNs Top Plays A legacy of giving: When it comes to giving back Judy and Bryan Roub lead by example Staph infection can come from ocean water beach sand; Florida man gets it from sandbar Its Launch Day! Things to know about SpaceX Falcon Heavy launch from Kennedy Space Center Viera 6 more Brevard HS softball teams enter playoffs Enjoy Jack Daniels tastings in Cocoa Village and Melbourne this month Brevard County location comes in at No 7 on list of US cities that get the most sun Letters and feedback: Oct 31 2019 Vietnam veteran receives settlement from Veterans Affairs  after waiting 20 years Brevard restaurants with outdoor seating wage war against lovebugs Aldi adds beer sparkling wine and dog treats to its collection of Advent calendars If millennials choose socialism fine Just dont make this mistake Gardening: Adding a tree to your Florida landscape? Southern redcedar is a great option Big Brevard high school football rivalries vary in postseason implications 6 hurricanes and 12 named storms predicted for remainder of 2019 hurricane season Cocoa man suspected of firing shots with unwitting mother in tow faces multiple charges Dogs on the beach? County Commission rejects proposal to allow dogs on South Brevard beaches Letters and feedback: Sept 15 2019 Despite weather odds SpaceX launches Falcon 9 rocket from Cape Canaveral Cyberstalking of alleged battery victim lands Rockledge man back in jail Winds are coming back again but inshore fishing is excellent for redfish and trout Exclusive: Trump administration sets plans for 2019 hurricane season after wakeup call of recent disasters Melting Pot: Intimacy of a shared meal makes this Viera spot perfect for date night The Chefs Table: Fine meat seafood draw diners to this Suntree favorite Holmes nurse arrested accused of sexual battery on a patient Tour of Brevard: Satellite Scorpions Djons Steak & Lobster House: Discover old-Florida elegance in Melbourne Beach restaurant Judicial Nominating Commission chair resigns claims governors office interfered with independence Medical marijuana firms seek to open more storefronts in Florida Brevard School votes 4-1 to support superintendents plan for teacher raises; audience not happy Tracking Brevards stars: Arena football champ UCF star PBA rookie lead the way Brevard school district miscalculated attrition savings finds $15 million for teacher pay Brevard high school cross country honor roll Oct 31 NASA Kennedy Space Center team wins Emmy for coverage of SpaceX mission Golf tip: Making adjustments to correct your swing Cruise ship rescues and mishaps: 6 times emergency struck on Royal Caribbean Carnival more Updates: SpaceX launches Falcon 9 from Cape Canaveral Air Force Station Secular group challenges In God We Trust motto on Brevard sheriffs patrol vehicles Chemicals in sunscreen seep into your bloodstream after just one day FDA says Cottage Rose mural dispute resolved in Indialantic after owner applies for permit Humberto strengthens into a hurricane as it moves away from Space Coast; NHC watching 2 more Planting herbs to cutting back on water heres what to do in your Brevard yard this month Viera Suntree Little League team heads to Michigan for Junior World Series We must prevent human trafficking through education | Opinion Please dont eat the deer Family spots Florida panther wandering around Estero backyard Man found hanging onto side of kayak in Atlantic Ocean off Patrick Air Force Base SpaceX set to launch Falcon Heavy on mission with 24 payloads â€“\xa0including human ashes Erin Baird is 2019 Volunteer of the Year finalist in FLORIDA TODAY Volunteer Recognition Awards ZZ Top Steve Martin and Martin Short coming to King Center Needy Brevard students sleep easier on new beds donated from Ashley HomeStore These fitness trends are worth sticking with over time Florida saw fewer infections from Vibrio vulnificus (aka flesh-eating bacteria) in 2018 Health calendar: April 25-May 2 Young actresses shine in Matilda at Titusville Playhouse Freshman quarterback Gabriel leads UCF over Stanford in Orlando John Daly is 2019 Volunteer of the Year finalist in FLORIDA TODAY Volunteer Recognition Awards Health Pro: Doctors interest grew out of childhood health issues More strong storms move eastward over Brevard; funnel clouds possible in Viera Nurse at Holmes Regional Medical Center arrested on suspicion of sexual battery Strong isolated showers heat impacting Brevard County Brevard firefighters widow wins $9 million verdict in Dominos Pizza delivery death Housing providers may be prohibited from rejecting renters on basis of arrest record 10-foot great white shark Miss May pings off Satellite Beach Palm Bay man 22 dies in overnight crash on Floridas Turnpike Tamara Carroll is 2019 Volunteer of the Year finalist in FLORIDA TODAY Volunteer Recognition Awards Tropical Storm Humberto not expected to affect Space Coast; NHC watching two more disturbances Theater veteran Blackledge passes away unexpectedly Ten dead issues from the 2019 legislative session Letters and feedback: June 23 2019 Rockledge Cocoa win big as Space Coast takes Holy Trinity in OT Surfers can enjoy chest-high waves most of weekend in Brevard Plan ahead for Aug 14-20: Space Coast Symphony performs Mikado Chef Larry tweaks hours Daylight saving time ends on Nov 3 Here are 7 facts to know about the time change Two new Royal Caribbean ships including second largest in world move to Port Canaveral Fla Sen Debbie Mayfield files bill to define e-cigarettes liquid nicotine as tobacco products Escape the hustle and bustle for a western getaway Missing Palm Bay teen found in South Carolina; mother arrested police say Impossible burger Impossible Cottage Pie make 2019 Epcot Food and Wine Festival menu Got bats? Why you need to get rid of them by Tax Day 2 charged with attempted murder; Titusville police say they pistol-whipped waterboarded force-fed him dog food Bicyclist dies in hit-and-run crash on US 1 near Cocoa Group: St Johns River pollution threatens water supplies Rockledge man charged after hoax to shoot up Merritt Island Walmart Man awarded $80M in lawsuit claiming Monsantos Roundup causes cancer Astronauts Alisa Rendina won FLORIDA TODAY Community Credit Union Athlete of the Week vote Alligators and airboats: How to explore Everglades National Park in one day Longtime educator church mother and humanitarian Johnnie Mae Riley remembered in Palm Bay George Clooney astronauts to attend black-tie event at Kennedy Space Center High school sports results: March 27 Bill Nye visits Cocoa Beach ahead of SpaceX Falcon Heavy launch from Kennedy Space Center Health First to relocate Cape Canaveral Hospital to Merritt Island open new facilities Letters and feedback: Aug 7 2019 Duran to host top area amateur golfers this weekend Buckaroo Ball benefits Harmony Farms Get ready to rumble: Atlas V blasting off early Thursday morning War on lovebugs: Social media reactions tweets memes about pesky bugs cars windshields Master association that contains condominiums may be able to collect capital contribution The new LEGO Movie World opens at LEGOLAND River Rocks: Enjoy fresh seafood and river views at this Rockledge tradition March 27: Panthers Walk It Off 10 takeaways from the presidents campaign kickoff visit to Orlando Cocoa man arrested on manslaughter charges after Sharpes shooting Corrections & Clarifications Accused burglars hit five south Brevard stores before Melbourne officer shot one of them Brevard family to receive Filipino World War II veterans Congressional Gold Medal NASA: Want to go to the moon in five years? We need more money How to REALLY clean lovebugs off your windshield car house Top party schools: Florida Florida State make it to top of Princeton Reviews 2020 list Live scores: High school football Week 4 in Brevard County Restaurant inspections: Happy Kitchen in West Melbourne Divine Grace in Palm Bay closed after inspections School district sticks to $770 raise offer for teachers; special magistrate getting involved Candidate lineup is set for 2019 municipal special district elections in Brevard Death investigation underway after mans body found in Titusville ditch November hurricane forecast: Season not technically over but its over | WeatherTiger Tyndall Air Force Bases future remains uncertain after devastation from Hurricane Michael Whats behind a low performing school? Crummy parents | Opinion Well known Chevrolet dealership has The Right Stuff Lovebugs are back Many hate them Experts say theyre here for good The lax disciplinary policies that caused\xa0Parkland massacre may have spread to your school Port Canaveral commissioners tackle parking issues: 10 things to know Melbourne woman injured in crash dies from her injuries; police looking for witnesses Senate President Galvano calls for Florida lawmakers to focus on mass shootings Governor DeSantis approves $91 billion budget slashes $133 million in local projects Letters and feedback: March 28 2019 Vote for 321prepscom Community Credit Union Athlete of the Week for April 29-May 4 South Patrick Shores approved for federal cleanup program Island H2O Live! water park in Kissimmee opens: Make the plunge or chill in lazy river Travis Proctor is 2019 Citizen of the Year finalist in FLORIDA TODAY Volunteer Recognition Awards Vietnam War 50th anniversary to be remembered at Cape Canaveral National Cemetery Its launch day! Things to know about SpaceX Falcon 9 launch from Cape Canaveral NASA gave Artemis contractors bonuses despite delays and cost overruns GAO says Michael Bloom is 2019 Citizen of the Year finalist in FLORIDA TODAY Volunteer Recognition Awards Health Pro: Eye doctor honed skills on aircraft carrier Brevard pitchers fare well in state horseshoe event The real reason NASAs all-female spacewalk is not happening Hint: Its not discrimination Port Canaveral aquarium the best use of tourism dollars | Opinion SpaceX Falcon Heavy will carry worlds first green rocket fuel mission School District union at odds on whats actually available for teacher raises | Opinion Nuclear power is too costly and too risky Jarvis Wash is 2019 Citizen of the Year finalist in FLORIDA TODAY Volunteer Recognition Awards SpaceX Dragon installed at ISS after launch from Cape Canaveral Rare phenomenon: Hail damage from unusual thunderstorm has Brevard residents talking Signs you live on the Space Coast Titusville couple charged with neglect after infant found malnourished Health coaches a personal way to help you meet goals New Viera elementary school to break ground in fall fast-tracked to open August 2020 Phase out nuclear power? Elizabeth Warren and Bernie Sanders locked in absurd arms race Strange days: Space Coast gets freak hail flaming meteor mysterious rumble red tide over 6 months Four women arrested on prostitution charges in ongoing Melbourne enforcement detail Whose name is on deed - partner or spouse? It matters a lot Florida should steer clear of drug importation from other countries | Opinion Melbourne police investigate Palm Bay Road shooting Brevard bowlers head to state tournament off district wins Florida Historical Societys May 16-18 conference: Countdown to History: Ice Age to the Space Age Turning the Toxic Tide: Florida must address the problem of human waste Contributions pour in to group seeking to ban possession sale of assault weapons in Florida Why Disney World sweets treats desserts are SO worthy of Instagram Satellite Beach girl 10 leads charge for organ donation Poliakoff: Community development district offers different advantages and disadvantages to an HOA Florida Tech in Melbourne hosts International Dinner with food from South Latin America From discovering water to snapping selfies: The lasting memories of Mars rover Opportunity Charges dropped against Texas man after comments against Cocoa Beach hotel werent direct specific threat Brevard Federation of Teachers just cant say yes to raise offers | Opinion How Mars Opportunity rover ranks in list of more than 1000 unmanned space missions by NASA since 1958 321 Flavor members rave about Fresh Scratch Bistro Sushi Factory and Beachside Seafood Texas man recovering after shark bite in Florida Buckeye the manatee rescued again by SeaWorld Rain hail pelts north central Brevard County causing widespread damage Regulators approve merger of Harris L3; deal to close June 29 Teachers approve largest pay raise in years but will it keep them in Brevard? Restaurant review: Continental Flambe in Melbourne shows great promise Weak cool front could bring torrential rain damaging winds to Space Coast Officials: Removing Beachline earthen causeways would improve Banana River water quality Buzz Aldrins outfit is everything in photo of last remaining Apollo astronauts Voting Republican only helps if youre rich | Opinion Homelessness: A national problem with a local solution EPA plans to regulate cancer-causing chemicals found in Americas drinking water UCF has chance to wow recruits playoff voters with a win over Stanford Ping-pong-sized hail pounds Space Coast as severe thunderstorms move through area Harris CEO optimistic about merger finances as deal with L3 nears completion March 26: EFSCâ€™s Carter Stewart named NJCAA Region VIII Pitcher of the Week Knife-wielding suspect leads deputies officers on foot chase in Melbourne 4 Brevard winners in girls regional basketball Thursday March 26: Womenâ€™s Golf Finishes Runner-Up at Barry Invite Some seek healing remembrance at arrival of Vietnam War replica wall High school sports results: March 26 Following surge in discipline problems Brevard students ask lawmakers to address vaping in schools Titusville man 21 pronounced dead after trying to make a U-turn on South Street Donald Trumps criminal justice reform shows results | Opinion Florida cold case solved 22 years later after Google Earth satellite image shows missing mans car Health calendar: Aug 8-15 Tracking Brevards stars: Harris to World Cup; Dawson in golf hunt; Allen Taylor hitting 300 Satellite Beach hotel condo towers homes slated for 27 acres by Hightower Beach park Letters and feedback: Sept 14 2019 Mike Martin Jr to be named Florida States new baseball coach Golf tip: Hit down on hybrids similar to how you hit an iron Stormy weather moves across Space Coast bringing lightning heavy rain Commission OKs pay raises for 625 county employees to address disparities turnover High school sports results: Feb 14 weVENTURE is 2019 Organization of the Year finalist in FLORIDA TODAY Volunteer Recognition Awards Man taken to hospital after domestic incident in Titusville Brevard school district considers selling $800000 in unused land Weather looks OK for SpaceX Falcon Heavy launch from Kennedy Space Center 321 Flavor: Margarita tops the list of Brevards favorite cocktails UCF experiment and Citronaut fly to space on Blue Origins New Shepard Bridges sex assault: Victimâ€™s family sues nonprofit caregiver over pregnancy Sheriff Ivey and Brevard emergency officials send mixed messages during Dorian New proposal on dog cat sales at pet stores is less restrictive than previous one Space Coast Honor Flight is 2019 Organization of the Year finalist in FLORIDA TODAY Volunteer Recognition Awards It goes by so quickly: Brevard teachers back to work as days count down to school County commissioners reject lagoon funding plan saying they want less money for muck projects Yuck! Hypodermic needles stabbing employees inside Brevard recycling plant Hard-line anti-illegal immigration group works to sway sanctuary cities bill Titusville High School principal retires amid accusations of hostile environment Pol Potâ€™s deputy Nuon Chea architect of bloody Khmer Rouge policy dead at 93 in Cambodia NASAs new goal: Land astronauts on the moon by 2024 Going surfing this weekend? Check our forecast Neighbor Up is 2019 Organization of the Year finalist in FLORIDA TODAY Volunteer Recognition Awards Severe storms bring wind lightning and possible hail across Brevard Starbucks in Indialantic West Melbourne may move to larger stores with drive-thru windows SpaceX Falcon Heavy: Why everyone should watch this night launch from Kennedy Space Center Fishing report: Calm seas and good fishing ahead Two World Series commercials feature Brevard celebrities Letters and feedback: May 5 2019 In latest motion US Attorney confirms multiple investigations in Tallahassee Lane 70 softball team wins spring tournament Palm Bay police double down on nightclub safety following four violent episodes Relativity Space signs deal to deliver small satellites to more orbits HBO documentary new surf movie spotlight surf legends Kelly Slater Hobgood twins Plan ahead for April 3-10: Music in Melbourne adult prom in Titusville; art in Eau Gallie Family grieves woman fatally struck by car near Lous Blues bar in Melbourne At Cape Canaveral SpaceX kicks off Star Wars Day with smoke and fire Updates: SpaceX launches Falcon 9 from Cape Canaveral to ISS sticks drone ship landing Letters and feedback: March 27 2019 Florida Tech to host event celebrating Apollo 11 anniversary and future of US space program Boeing on Starliner orbital flight test: We really are close Florida Tech spring football game set for Saturday Feb 14: EFSC womenâ€™s golfer Kacey Walker nominated for 2019 Arnold Palmer Cup Four Panthers Selected to All-SSC Teams Heavy winds and lightning to hit north Brevard County How Ron DeSantis is shaking up the establishment | Opinion High school sports results: May 3 Palm Bay man who asked woman for permission to have illicit sexual encounter with child sentenced Letters and feedback: June 22 2019 Air Force pumped for two launches less than 48 hours apart Heres why NASA scrubbed all-female spacewalk Plan ahead for Nov 6-12: Holiday bazaar in Suntree pet events art openings and food fun Edgewoods Jon Wang shoots one under par for second time this week Burglar steals bolted safe from Melbourne bar in latest smash-and-grab Feb 14: Florida Tech Suffers Last-Second Heartbreaker Against Saint Leo Daddy Duty: My school cheating scandal is an example for my 4-year-old Tropical Storm Humberto forms but isnt expected to affect Space Coast Air Force weather squadron key player in ensuring successful rocket launches Ronsisvalle: Getting through to the disconnected husband How to watch SpaceX launch its Falcon 9 rocket from Cape Canaveral Air Force Station Members of Brevards Bahai Faith community come together to commemorate twin holy days A&Es Tiny House Nation to feature Melbourne tiny home builder Movable Roots Lawmakers grapple with felons voting rights All-female spacewalk canceled due to lack of medium-sized spacesuits Twitter reacts Medical marijuana and 3 other school district policy changes you should know about Felons rights bill heads to Gov DeSantis Lanes open delays persist after Crash on I-95 near Wickham Road Former UCF chair resigns day after Rep Randy Fine proposes UC'
# creating spacy doc
news_doc=nlp(news_articles)

# creating an empty list to store people's names
list_of_people=[]

# iterating through every token in doc
for token in news_doc:
  # checking if category matches
  if token.ent_type_=='PERSON':
    # appending to list if PERSON
    list_of_people.append(token.text)

print(list_of_people)
```

You can iterate through the entities present in the doc using ```.ents``` and print their label through ```.label_```
```
# Print named entities
for entity in doc.ents:
  print(entity.text,'--',entity.label_)
```

Sometimes, you want to hide names of people, companies etc
```
# Function to identify  if tokens are named entities and replace them with UNKNOWN
def remove_details(word):
    if word.ent_type_ =='PERSON' or word.ent_type_=='ORG' or word.ent_type_=='GPE':
        return ' UNKNOWN '
    return word.string


# Function where each token of spacy doc is passed through remove_deatils()
def update_article(doc):
    # iterrating through all entities
    for ent in doc.ents:
        ent.merge()
    # Passing each token through remove_details() function.
    tokens = map(remove_details,doc)
    return ''.join(tokens)

# Passing our news_doc to the function update_article()
update_article(news_doc)
```


### Useful functions
```
import string
import unidecode
from nltk.tokenize import word_tokenize


DICT_REPLACE = {"-" : " ",
                "'" : " "}
def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda x: dict[x.string[x.start():x.end()]], text)

def preprocess_data(df):
    # remove punctuation
    df['text_new'] = df['script'].apply(lambda x: ' '.join(word.strip(string.punctuation) for word in x))
    # lower case
    df['text_new'] = df['text_new'].apply(lambda x: x.lower())
    # replace characters
    df['text_new'] = df['text_new'].apply(lambda x: self.multiple_replace(DICT_REPLACE, x))
    # Remove accent
    df['text_new'] = df['text_new'].apply(lambda x: unidecode.unidecode(str(x)))
    # tokenization
    df['text_new'] = df['text_new'].apply(lambda x: word_tokenize(x))
    return df

```
