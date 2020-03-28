/**
 * @file   Decoder.h
 * @author Raffaele Montella <raffaele.montella@uniparthenope.it>
 * @date   Wed Nov 30 17:16:48 2016
 *
 * @brief
 *
 *
 */

#pragma once

#include <cstdlib>
#include <iostream>

#include <log4cplus/configurator.h>
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

namespace gvirtus::common {
typedef enum { step_a, step_b, step_c, step_d } DecodeStep;

class Decoder {
 public:
  Decoder();

  Decoder(const Decoder &orig);

  virtual ~Decoder();

  int Decode(char value_in);

  int Decode(const char *, int, char *);

  void Decode(std::istream &, std::ostream &);

 private:
  log4cplus::Logger logger;
  int _buffersize;
  DecodeStep step;
  char plainchar;

  int Value(char value_in);

  int Block(const char *code_in, const int length_in, char *plaintext_out);
};
}  // namespace gvirtus::common
