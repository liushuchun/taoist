set(cnn_library_SRCS dim.cpp
        tensor.cpp
        )

set(cnn_library_HDRS
        dim.h
        tensor.h)

add_library(cnn STATIC ${cnn_library_SRCS} ${cnn_library_HDRS})

install(FILES ${cnn_library_HDRS} DESTINATION include/cnn)
install(TARGETS cnn DESTINATION lib)