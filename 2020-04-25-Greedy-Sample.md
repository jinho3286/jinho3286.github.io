---
layout: single
title: "Greedy Sample"
date: 2020-04-25 14:00:00 +0900
authour: Jinho
---

# Greedy Sample



각종 코딩 대회, 사이트 등을 돌아다니다가 딱히 신박한(?) 문제도 , 코드도 보이지 않아서 한참 실망하고 있었다. 그나마 끌렸던 주제는 프로그래머스에서 제공한 문제인 "고속도로 카메라 설치하기" 였는데, 코딩 대회에 올라온 주제여서 그런지 새로운 느낌은 그닥 들지 않았다. 뭐 나머지 코드들도 어디서 어떻게 쓰이냐에 따라 실생활에서 유용하게 쓰이겠지만 내가 이번 케이스를 가져온 건 실생활에서 쓰이고 있기 때문에 신박하다고 느껴서 조사하게 됬다.



### RSS feed

---

`RSS(Really Simple Syndication / Rich Site Summary)` 같은 "사이트 피드"란, 새 기사들의 제목만, 또는 새 기사들 전체를 뽑아서 하나의 파일로 만들어 놓은 것이다.



때문에 각 사이트들에서 제공하는 `RSS파일 주소`만 수집하여 확인하면, 자신의 취향에 맞는 새로운 읽을거리를 쉽게 찾아서 읽을 수 있다.



##### 그럼 왜 쓸까?

* 기존 각 사이트에서 제공하는 서비스로 ,  개인 이메일로 새로운 내용을 보내주는 경우가 많다. 그러려면 자신의 메일 주소를 알려주어야 하는데 만약 그 사이트가 믿을만한 사이트가 아니라면 메일 및 개인 정보가 엉망이 될 것이다. 이를 방지하면서 원하는 정보를 알림 받고 싶을때 유용한 기능이다.
* 하지만 이보다 먼저 각 검색엔진의 수집봇이 최신 글을 수집하게끔 도와주는 역할로 활발히 쓰였다고 한다. 일종의 pull기능보다 push기능을 위해 먼저 쓰였다고 할까?



#### 사용법

---

이왕 조사한김에 쓸 수 있는 방법도 알면 좋을 것 같아서 같이 조사를 해 보았다. 

[RSS사용방법]([https://maternalgrandfather.tistory.com/entry/%ED%8B%B0%EC%8A%A4%ED%86%A0%EB%A6%AC-RSS-%ED%94%BC%EB%93%9C-%EB%9C%BB-%EC%84%A4%EC%A0%95%EB%B0%A9%EB%B2%95](https://maternalgrandfather.tistory.com/entry/티스토리-RSS-피드-뜻-설정방법))

* 각각 OS에 알맞는 RSS리더기 설치
* 원하는 블로그 , 사이트 피드주소 넣기

ex)  `https://maternalgrandfather.tistory.com/rss`

* RSS리더기에서 내가 아지 읽지 않은 최신글 알람 해줌



쉽죠? :D



#### 그럼 왜?

---

* `매 순간` 홈페이지, 블로그에서 올라오는 포스팅을 캐치 해 `알려주는`  방법을 사용하고 있으므로  `Greedy` 라고 판단했다.
* `중요도`를 고려한 것이 아니라 `그 즉시` `이득이 되는대로` 판단하기 때문이랄까?



#### 때마침

---

[로저 형님](https://github.com/rogierlommers/greedy)이 RSS reader기를 만들어 보셨는지 repository를 공유하셨길래 코드를 좀 살펴보았다.  [READ.me](https://github.com/rogierlommers/greedy/blob/master/readme.md) 파일에 너무 설명을 잘 해 놓으셔서 몰랐던 용어들 빼고는 읽어보기 편했다.



* `databasefile` 환경을 설정 함으로써 받아온 주소 저장장소를 관리 할 수 있단다.

```
docker run -v /srv/services/greedy:/greedy-data -p 9001:8080 --name greedy rogierlommers/greedy
```

port 8080 에서 운영되는 것을 9001로 저장함으로써 local backup

을 자신이 지정한 `directory`에 저장 할 수 있단다.

그런식으로 `host:port`  처럼 RSS Reader에 블로그 내용을 불러 올 수 있나보다.



![](https://github.com/rogierlommers/greedy/raw/master/docs/gui-02.png) 

오 신기하네~



이제 짤막하게 코드를 살펴보자



#### 코드 [코드출처](https://github.com/rogierlommers/greedy)

---

* 메인코드

```java
package main

import (
	"fmt"
	"net/http"

	log "github.com/Sirupsen/logrus"
	"github.com/gorilla/mux"
	"github.com/rogierlommers/greedy/internal/articles"
	"github.com/rogierlommers/greedy/internal/common"
	"github.com/rogierlommers/greedy/internal/render"
)

func main() {
	// read environment vars and setup http client
	common.ReadEnvironment()
	articles.NewClient()

	// initialize bolt storage
	articles.Open()
	defer articles.Close()

	// initialize mux router
	router := mux.NewRouter()

	// selfdiagnose
	common.SetupSelfdiagnose()

	// setup statics
	render.CreateStaticBox()

	// http handles
	router.HandleFunc("/", articles.IndexPage)
	router.HandleFunc("/add", articles.AddArticle)
	router.HandleFunc("/rss", articles.DisplayRSS)
	router.HandleFunc("/export", articles.ExportCSV)

	// schedule cleanup routing
	articles.ScheduleCleanup()

	// start server
	http.Handle("/", router)
	log.Infof("deamon running on host %s and port %d", common.Host, common.Port)

	err := http.ListenAndServe(fmt.Sprintf("%s:%d", common.Host, common.Port), nil)
	if err != nil {
		log.Panicf("daemon could not bind on interface: %s, port: %d", common.Host, common.Port)
	}
}
```

각각의 메소드들을 실행만 시키는 메인창이라 디테일한 내용은 볼 수 없지만 보통  메소드의 기능을 중심으로 이름을 짓기에 대충 "아 ~ 이런 기능들을 집어넣었구나 " 하고 보기 좋을거 같아서 가져왔다.



```java
func DisplayRSS(w http.ResponseWriter, r *http.Request) {
	now := time.Now()
	feed := &feeds.Feed{
		Title:       "your greedy's personal rss feed",
		Link:        &feeds.Link{Href: common.FeedsLink},
		Description: "Saved pages, all in one RSS feed",
		Author: &feeds.Author{
			Name:  common.FeedsAuthorName,
			Email: common.FeedsAuthorEmail,
		},
		Created: now,
	}

	db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte(bucketName)).Cursor()
		count := 0
		for k, v := c.Last(); k != nil; k, v = c.Prev() {
			if count >= numberInRSS {
				break
			}

			var a *Article
			a, err := decode(v)
			if err != nil {
				return err
			}

			newItem := feeds.Item{
				Title:       a.Title,
				Link:        &feeds.Link{Href: a.URL},
				Description: a.Description,
				Created:     a.Added,
				Id:          strconv.Itoa(a.ID),
			}
			feed.Add(&newItem)
			count++
		}

		// update stats
		s.setLastCrawler(r.UserAgent())
		s.incCrawlCount()
		return nil
	})

	rss, err := feed.ToAtom()
	if err != nil {
		log.Errorf("error while generating RSS feed: %s", err)
		return
	}
	w.Write([]byte(rss))
}
```

**일부  코드**만 가져와보았다.

* 현재 시간에 업로드 된 주소를 잡아와  가시화 시켜주는 코드인 것 같다. 중간에 count도 들어가 있는걸 보니 내가 아직 읽지 않은 기사가 몇개인지도 알려주려고 짜놓았나보다.
* 코드 자체는 `Greedy Algorithm`틱한 것 같진 않다. 기능상 `Greedy`하다그래야 할까? 

##### Idea?

---



* 조금 코드를  추가해보면 , 이 Reader기에 `업로드 된`기사들 중에 내가 `우선적`으로 보고싶은 `단어`들에 가중치를 두어서 `업로드 된 시간이 늦`더라도 `가장 위에`올라오게 끔 하면 좀 더 `Greedy Algorithm`틱 한 코드가 되지 않을까 싶다.

---



```java
func AddArticle(w http.ResponseWriter, r *http.Request) {
	queryParam := r.FormValue("url")
	if len(queryParam) == 0 || queryParam == "about:blank" {

		renderObject := map[string]interface{}{
			"IsErrorPage":  "true",
			"errorMessage": "unable to insert empty or about:blank page",
		}
		render.DisplayPage(w, r, renderObject)
		return
	}

	newArticle := Article{
		URL:   queryParam,
		Added: time.Now(),
	}

	err := newArticle.Save()
	if err != nil {
		log.Warn("error saving article", "hostname", getHostnameFromUrl(queryParam), "id", newArticle.ID)
	}

	// finally output confirmation page
	renderObject := map[string]interface{}{
		"IsConfirmation": "true",
		"hostname":       getHostnameFromUrl(queryParam),
	}
	render.DisplayPage(w, r, renderObject)
}
```

* 아티클 주소 받아와 저장하기



```java
func ScheduleCleanup() {
	go func() {
		log.Infof("scheduled cleanup, every %d seconds, remove more than %d records", scheduleCleanup, keep)
		for {
			deleted := cleanUp(keep)
			log.Infof("deleted %d records from database", deleted)
			time.Sleep(scheduleCleanup * time.Second)
		}
	}()
}
```

* 그렇지, 내가 다 읽은것과 읽지않은것을 구분하려면 이 기능도 있어야 겠지



#### 느낀점

---

사실 그렇게 어려운 내용의 코드도 아니고 좀 더 깊이 코드를 만진 것 같지는 않다.  나는 코드의 난이도 보다는 오히려 가볍더라도 더욱 `실용적`이게 쓰이는 코드가 `더 좋은`코드 이지 않을까 라는 생각을 한다. 

실제로 유용하게 쓰이고 있는 분야를 알게 되어서 좋았고 , 이왕 알게된 김에 `RSS Reader`를 이용해서 이제 `뉴스`좀 읽고 세상공부좀 해봐야겠다 하는생각이 들었다. 

* 좋은데 이거?