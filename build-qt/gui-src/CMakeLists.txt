cmake_minimum_required(VERSION 3.21)
project(CNNWorkbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Charts Concurrent)

add_library(cnn_core STATIC
    ../../Models.cpp
    AppBackend.cpp
)
target_include_directories(cnn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../
)

add_executable(cnn_gui
    main.cpp
    app_resources.qrc
)
target_link_libraries(cnn_gui PRIVATE cnn_core Qt6::Widgets Qt6::Charts Qt6::Concurrent)

add_executable(cnn_cli
    ../../cli_main.cpp
)
target_include_directories(cnn_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../
)
target_link_libraries(cnn_cli PRIVATE cnn_core)
