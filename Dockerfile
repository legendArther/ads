FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /home/user/app/cache
RUN chown -R user:user /home/user/app

USER user

EXPOSE 7860
CMD ["python", "app.py"]
