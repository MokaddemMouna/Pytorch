FROM gcr.io/google-appengine/python

COPY . .

RUN chmod +x run_server.sh

RUN pip3 install --upgrade pip

RUN pip3 install flask && \
    pip3 install gensim && \
    pip3 install torch && \
    pip3 install tqdm && \
    pip3 install cymem && \
    pip3 install nltk && \
    pip3 install pandas && \
    pip3 install numpy

EXPOSE 8080

ENTRYPOINT ["run_server.sh"]
CMD ["master"]