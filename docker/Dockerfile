FROM kitware/cmake:ci-clang_cxx_modules-x86_64-2023-02-15

RUN yum -y install which gmp-devel python3 python3-pip
RUN ln -s $(which python3) /usr/bin/python
RUN pip3 install toml loguru tqdm

ENV ELAN_HOME="/.elan"
ENV PATH="${ELAN_HOME}/bin:${PATH}"
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y

RUN chmod -R a+w /.elan

WORKDIR /workspace
RUN chmod -R a+w /workspace