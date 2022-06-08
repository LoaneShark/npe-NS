FROM bitnami/git AS fetcher
RUN git clone --progress https://github.com/deepchatterjeeligo/lalsuite.git && \
    cd lalsuite && git checkout eppe


FROM continuumio/miniconda3 AS builder
COPY --from=fetcher lalsuite lalsuite
RUN conda update --yes -n base -c defaults conda && \
    cd lalsuite && conda env create -f conda/environment.yml
RUN cd lalsuite && \
    conda run -n lalsuite-dev ./00boot
RUN cd lalsuite && \
    conda run -n lalsuite-dev ./configure \
        --prefix=/opt/conda/envs/lalsuite-dev \
        --enable-swig-python \
        --disable-lalstochastic \
        --disable-lalxml \
        --disable-lalinference \
        --disable-laldetchar \
        --disable-lalapps
RUN cd lalsuite && conda run -n lalsuite-dev make
RUN cd lalsuite && conda run -n lalsuite-dev make install

COPY . eppE/
RUN cd eppE && \
    conda run -n lalsuite-dev pip install .

ENTRYPOINT ["conda", "run", "-n", "lalsuite-dev"]