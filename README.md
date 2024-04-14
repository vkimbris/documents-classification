# Решение команды "Своего рода ученые" для хакатона от Цифрового прорыва

## Варианты взаимодействия с решением
1. Локально. Инструкция по установке ниже
2. Удаленно. Наше решение развернуто на YandexCloud
   
   UI: <ссылка на юай>\
   Backend Swagger UI: <ссылка на сваггер>

## Установка

1. Убедитесь, что у вас установлен `docker`. Для Linux: `sudo apt-get install docker.io`
2. Выполните команду `docker-compose up --build`
3. Откройте в браузере <ссылка на локальный юай>

## Описание методов
### GET /labels
Возвращает текущий список классов документов:
```json
{
  "labels": [
    "Акты",
    "Доверенности",
    "Договоры",
    "Договоры оферты",
    "Заявления",
    "Постановления",
    "Приказы",
    "Приложения",
    "Распоряжения",
    "Решения",
    "Соглашения",
    "Счета",
    "Уставы"
  ]
}
```

### POST /classifyDocuments
Принимает на вход список документов в любом текстовом формате. Возващает типы документов:
```json
[
  {
    "file": "Договоры_0a634971c9ce691123fb78752464d24d16d5bbabedd7b2f858a6c2cc.txt",
    "label": "Договоры"
  },
  {
    "file": "Заявления_4dec9edaaba74e6265d7a65c0c414621e750ee008ef00410b5ab8307.txt",
    "label": "Заявления"
  }
]
```

### POST /updateModel
Принимает на вход файл в формате .csv с двумя столбцами: "text" и "class". Модель будет переобучена с учетом предыдущих и новых данных.\
В случае успешного обучения, будут возвращены метрики качества обучения.
```json
{
  "status": "Model trained succesfully.",
  "classificationReport": {
    "Акты": {
      "precision": 1,
      "recall": 0.5714285714285714,
      "f1-score": 0.7272727272727273,
      "support": 7
    },
    "Доверенности": {
      "precision": 1,
      "recall": 1,
      "f1-score": 1,
      "support": 5
    },
  }
}
```

### POST /topicModeling
Метод для выделения скрытых топиков в документах, предоставленными организаторами.\
Принимает на вход число топиков, на которое необходимо разбить все документы. Возвращает JSON следующего формата:
```json
[
  {
    "topicID": 0,
    "topicKeyWords": [
      "компания",
      "элемент",
      "инспекция",
      "штраф",
      "прочие",
      "евразийский",
      "переработка",
      "кредитный",
      "пошлина",
      "покупатель"
    ],
    "topicRepresentativeDocuments": ["содержимое_документа_1", "содержимое_документа_2"]
  }
]
```
"topicID" - номер топика\
"topicKeyWords" - список ключевых слов топика\
"topicRepresentativeDocs" - список самых репрезентативных документов топика\

Для того, чтобы получить более человекочитаемое представление топиков, можно воспользоваться ChatGPT, используя следующий промпт:
```
У меня есть топик, содержащий следующие документы: \n{topicRepresentativeDocs}.
Топик описывается следующими ключевыми словами: {topicKeyWords}.
Опираясь на информацию выше, можешь дать короткое название топику?
```
