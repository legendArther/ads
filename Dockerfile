FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /home/user/app/assets
COPY assets/ /home/user/app/assets/

RUN chown -R user:user /home/user/app
USER user

EXPOSE 7860
CMD ["python", "app.py"]
