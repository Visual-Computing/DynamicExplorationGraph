// maybe implementing with cpp26's std::hive: https://lemire.me/blog/2026/08/02/how-fast-is-c26s-stdhive/ or
// boost::hub container
// https://www.boost.org/doc/libs/develop/doc/html/container/non_standard_containers.html#container.non_standard_containers.hub
// both cases keep deleted space free and do not move around later parts to fill them. Which means if you add 2mio and
// remove 1mio afterwards it still takes memory for 2mio
