---
layout: single
title: "Greedy Algorithm"
date: 2020-04-21 15:00:00 +0900
authour: Jinho
---

# OJS : Greedy Algorithm 





### I)

---

#### 문제 설명

언제나 최고만을 지향하는 굴지의 대기업 진영 주식회사가 신규 사원 채용을 실시한다. 인재 선발 시험은 1차 서류심사와 2차 면접시험으로 이루어진다. 최고만을 지향한다는 기업의 이념에 따라 그들은 최고의 인재들만을 사원으로 선발하고 싶어 한다.

그래서 진영 주식회사는, 다른 모든 지원자와 비교했을 때 서류심사 성적과 면접시험 성적 중 적어도 하나가 다른 지원자보다 떨어지지 않는 자만 선발한다는 원칙을 세웠다. 즉, 어떤 지원자 A의 성적이 다른 어떤 지원자 B의 성적에 비해 서류 심사 결과와 면접 성적이 모두 떨어진다면 A는 결코 선발되지 않는다.

이러한 조건을 만족시키면서, 진영 주식회사가 이번 신규 사원 채용에서 선발할 수 있는 신입사원의 최대 인원수를 구하는 프로그램을 작성하시오.

#### 입력 설명

첫째 줄에는 테스트 케이스의 개수 T(1 ≤ T ≤ 20)가 주어진다. 각 테스트 케이스의 첫째 줄에 지원자의 숫자 N(1 ≤ N ≤ 100,000)이 주어진다. 둘째 줄부터 N개 줄에는 각각의 지원자의 서류심사 성적, 면접 성적의 순위가 공백을 사이에 두고 한 줄에 주어진다. 두 성적 순위는 모두 1위부터 N위까지 동석차 없이 결정된다고 가정한다.

#### 출력 설명

각 테스트 케이스에 대해서 진영 주식회사가 선발할 수 있는 신입사원의 최대 인원수를 한 줄에 하나씩 출력한다.

#### 입력 예시 )

```
2
5
3 2
1 4
4 1
2 3
5 5
7
3 6
7 3
4 2
1 4
5 7
2 5
6 1
```

#### 출력 예시)

```
4
3
```





```java
import java.util.Scanner;

public class Main {


    private static void solve(){
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();


        for (int i = 0; i <a ; i++) {


            int b = sc.nextInt();
            int [] arr = new int[b+1];


            for (int j = 0; j <b ; j++) {
                int doc = sc.nextInt();
                int interview =sc.nextInt();
                arr[doc]=interview;
            }

            int cnt =1;
            int std = arr[1]; // doc 1등의 interview 성적


            for(int k=2; k<=b; k++)
            {
              if(std >= arr[k]) {
                  std = arr[k];
                  cnt++;
              }
            }

            System.out.println(cnt);

        }


    }







    public static void main(String [] args) {
            solve();
    }
}

```





### H)

---

#### 문제 설명

한 개의 회의실이 있는데 이를 사용하고자 하는 N개의 회의에 대하여 회의실 사용표를 만들려고 한다. 각 회의 I에 대해 시작시간과 끝나는 시간이 주어져 있고, 각 회의가 겹치지 않게 하면서 회의실을 사용할 수 있는 회의의 최대 개수를 찾아보자. 단, 회의는 한번 시작하면 중간에 중단될 수 없으며 한 회의가 끝나는 것과 동시에 다음 회의가 시작될 수 있다. 회의의 시작시간과 끝나는 시간이 같을 수도 있다. 이 경우에는 시작하자마자 끝나는 것으로 생각하면 된다.

#### 입력 설명

첫째 줄에 회의의 수 N(1 ≤ N ≤ 100,000)이 주어진다. 둘째 줄부터 N+1 줄까지 각 회의의 정보가 주어지는데 이것은 공백을 사이에 두고 회의의 시작시간과 끝나는 시간이 주어진다. 시작 시간과 끝나는 시간은 231-1보다 작거나 같은 자연수 또는 0이다.

#### 출력 설명

첫째 줄에 최대 사용할 수 있는 회의의 최대 개수를 출력한다.

#### 입력 예시)

```
11
1 4
3 5
0 6
5 7
3 8
5 9
6 10
8 11
8 12
2 13
12 14
```

#### 출력 예시)

```
4
```



```java
import java.util.Scanner;

public class Main {


    private static void solve(){
        Scanner sc = new Scanner(System.in);
        int a= sc.nextInt();
        int [] arr1 =new int[a];
        int [] arr2 =new int[a];

        for (int i = 0; i <a ; i++) {
                int st = sc.nextInt();
                int fin = sc.nextInt();
                arr1[i] =st;
                arr2[i] =fin;
        }

        int tmp =0;
        int cnt =1;
        int std;
        int max=arr1[0];

        for (int i = 0; i <a ; i++) {
            if(max <=arr1[i])
                max=arr1[i];
        }

           while(true)
            {
            std =arr2[tmp];

            for (int i = 0; i <a ; i++) {
                if(std <=arr1[i])
                {

                    cnt++; tmp=i; break;
                }

            }
            if(max <std)
                break;

        }
        System.out.println(cnt);
    }






    public static void main(String [] args) {
            solve();

    }
}

```

