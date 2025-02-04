### Query movies
```sql
SELECT ?film ?filmLabel
       (GROUP_CONCAT(DISTINCT ?directorLabel; separator=", ") AS ?directors)
       (GROUP_CONCAT(DISTINCT ?producerLabel; separator=", ") AS ?producers)
       (GROUP_CONCAT(DISTINCT ?screenwriterLabel; separator=", ") AS ?screenwriters)
       (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") AS ?genres)
        (MIN(YEAR(?releaseDate)) AS ?release_year)
       (GROUP_CONCAT(DISTINCT ?countryLabel; separator=", ") AS ?countries_origin)
        (GROUP_CONCAT(DISTINCT ?locationLabel; SEPARATOR=", ") AS ?locations)
       (GROUP_CONCAT(DISTINCT ?durationLabel; separator=", ") AS ?durations)
WHERE {
  {
    SELECT ?film (MIN(YEAR(?releaseDate)) AS ?minYear)
    WHERE {
      ?film wdt:P31 wd:Q11424;
            wdt:P577 ?releaseDate.
    }
    GROUP BY ?film
    # Filter for films released in or after a specific year (e.g., 2000)
    HAVING(?minYear >= 2012 && ?minYear < 2013)
  }
  
  ?film wdt:P31 wd:Q11424;
        wdt:P57 ?director;
        wdt:P58 ?screenwriter;
        wdt:P136 ?genre;
        wdt:P577 ?releaseDate;
        wdt:P495 ?country;
        wdt:P2047 ?duration;
        wdt:P915 ?location;
        wdt:P162 ?producer.
        
  
  #VALUES ?genre {wd:Q188473}
  # Filter for films with English labels
  FILTER(LANG(?filmLabel) = "en")
  
  SERVICE wikibase:label { 
    bd:serviceParam wikibase:language "en".
    ?film rdfs:label ?filmLabel.
    ?director rdfs:label ?directorLabel.
    ?screenwriter rdfs:label ?screenwriterLabel.
    ?genre rdfs:label ?genreLabel.
    ?country rdfs:label ?countryLabel.
    ?duration rdfs:label ?durationLabel.
    ?location rdfs:label ?locationLabel.
    ?producer rdfs:label ?producerLabel.
  }
}
GROUP BY ?film ?filmLabel
ORDER BY ?earliestReleaseYear
```
### Query movies (v2)
```
SELECT ?film ?filmLabel
       (GROUP_CONCAT(DISTINCT ?directorLabel; separator=", ") AS ?directors)
       (GROUP_CONCAT(DISTINCT ?screenwriterLabel; separator=", ") AS ?screenwriters)
       (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") AS ?genres)
       (MIN(YEAR(?releaseDate)) AS ?release_year)
       (GROUP_CONCAT(DISTINCT ?durationLabel; separator=", ") AS ?durations)
(GROUP_CONCAT(DISTINCT ?castLabel; separator=", ") AS ?cast_list)
WHERE {
  ?film wdt:P31 wd:Q11424;
        wdt:P57 ?director;
        wdt:P58 ?screenwriter;
        wdt:P136 ?genre;
        wdt:P577 ?releaseDate;
        wdt:P161 ?cast;
        wdt:P2047 ?duration.
  
  # Filter for films with English labels and released after 2000
  FILTER(LANG(?filmLabel) = "en")
  FILTER(YEAR(?releaseDate) = 2014)
  
  SERVICE wikibase:label { 
    bd:serviceParam wikibase:language "en".
    ?film rdfs:label ?filmLabel.
    ?director rdfs:label ?directorLabel.
    ?screenwriter rdfs:label ?screenwriterLabel.
    ?genre rdfs:label ?genreLabel.
    ?duration rdfs:label ?durationLabel.
    ?cast rdfs:label ?castLabel.
  }
}
GROUP BY ?film ?filmLabel
ORDER BY ?release_year
LIMIT 800
````

### Query cities (low population)
```sql
SELECT DISTINCT ?city ?cityLabel ?countryLabel ?loc ?recent_population ?above_sea_meters  WHERE {
  {
    SELECT ?city (MIN(?population) AS ?recent_population) (MIN(?timezone) AS ?recent_timezone) (MIN(?above_sea) AS ?above_sea_meters)
    WHERE {
       ?city wdt:P31/wdt:P279* wd:Q515 ;
       wdt:P1082 ?population;
       wdt:P421 ?timezone;
       wdt:P2044 ?above_sea.
    }
    GROUP BY ?city
    HAVING(?recent_population >= 500 && ?recent_population <= 15000)
  }
  ?city wdt:P31/wdt:P279* wd:Q515 .
  ?city wdt:P1082 ?population .
  ?city wdt:P17 ?country .
  ?city wdt:P625 ?loc .
  ?city wdt:P421 ?timezone.
  ?city wdt:P2044 ?above_sea.
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
  }
}
```

### Query cities (high population)
```sql
SELECT DISTINCT ?city ?cityLabel ?countryLabel ?loc ?recent_population ?above_sea_meters  WHERE {
  {
    SELECT ?city (MIN(?population) AS ?recent_population) (MIN(?timezone) AS ?recent_timezone) (MIN(?above_sea) AS ?above_sea_meters)
    WHERE {
       ?city wdt:P31/wdt:P279* wd:Q515 ;
       wdt:P1082 ?population;
       wdt:P421 ?timezone;
       wdt:P2044 ?above_sea.
    }
    GROUP BY ?city
    HAVING(?recent_population >= 50000)
  }
  ?city wdt:P31/wdt:P279* wd:Q515 .
  ?city wdt:P1082 ?population .
  ?city wdt:P17 ?country .
  ?city wdt:P625 ?loc .
  ?city wdt:P421 ?timezone.
  ?city wdt:P2044 ?above_sea.
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
  }
}
```

### Query players (basketball)
```sql
SELECT ?human ?humanLabel (YEAR(?datebirth) AS ?datebirthyearLabel) ?placebirthLabel (GROUP_CONCAT(DISTINCT ?teamsLabel; separator=", ") AS ?teamslist) (MIN(?mass) AS ?recent_mass) (MIN(?height) AS ?recent_height)
WHERE {
  {
    SELECT DISTINCT ?human
    WHERE {
      ?human wdt:P31 wd:Q5 ;
             wdt:P106 wd:Q3665646;
             wdt:P54 ?teams;
             wdt:P569 ?datebirth;.
    }
    GROUP BY ?human
    HAVING (COUNT(DISTINCT ?teams) > 0 && COUNT(DISTINCT ?datebirth) > 0)
    LIMIT 15000
  }
  ?human wdt:P31 wd:Q5 ;
         wdt:P106 wd:Q3665646;
         wdt:P21 ?gender;
         wdt:P569 ?datebirth;
         wdt:P19 ?placebirth;
         wdt:P27 ?country_citizen;
         wdt:P54 ?teams;
         wdt:P2067 ?mass;
         wdt:P2048 ?height.
  
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
    ?human rdfs:label ?humanLabel.
    ?placebirth rdfs:label ?placebirthLabel.
    ?country_citizen rdfs:label ?country_citizenLabel.
    ?teams rdfs:label ?teamsLabel.
    ?mass rdfs:label ?massLabel.
    ?height rdfs:label ?heightLabel.
  }
}
GROUP BY ?human ?humanLabel ?datebirth ?placebirthLabel ?recent_mass ?recent_height
```

### Query players (song)
```sql
#Songs with performers and publication year
SELECT ?song ?songLabel ?albumLabel (YEAR(?publicationDate) AS ?publicationYear) (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") AS ?genres) (GROUP_CONCAT(DISTINCT ?performerLabel; separator=", ") AS ?performers)
WHERE
{
  ?song wdt:P31 wd:Q134556;
        wdt:P136 ?genre;
        wdt:P175 ?performer;
        wdt:P577 ?publicationDate;
        wdt:P361 ?album.
  
  SERVICE wikibase:label { 
    bd:serviceParam wikibase:language "en". 
    ?song rdfs:label ?songLabel.
    ?genre rdfs:label ?genreLabel.
    ?album rdfs:label ?albumLabel.
    ?performer rdfs:label ?performerLabel.
  }
}
GROUP BY ?song ?songLabel ?albumLabel ?publicationDate
LIMIT 10000
```