Сборка проекта:

``docker build -t recognizer .``

Запуск проекта:

``docker run -p 5555:5555 --network host recognizer:latest``