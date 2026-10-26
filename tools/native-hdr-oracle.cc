// Test-only oracle for Cloudinary SSIMULACRA2 2.1.
// Color/luminance helpers follow bundled libjxl, see fast-ssim2/LICENSE.libjxl.
// Uses upstream's own CMS and XYB conversion, without file decoding.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <lcms2.h>
#include <lcms2_plugin.h>

#include "lib/jxl/enc_color_management.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/linalg.h"
#include "lib/jxl/color_management.h"
#include "ssimulacra2.h"

static float Float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
static uint32_t Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static jxl::ImageBundle Read(jxl::ImageMetadata* metadata, float* bg, bool gray = false) {
  int transfer, primaries, alpha;
  size_t width, height;
  float target;
  if (!(std::cin >> transfer >> primaries >> target >> *bg >> alpha >> width >> height))
    throw std::runtime_error("missing image header");
  jxl::ColorEncoding color;
  if (!color.SetSRGB(gray ? jxl::ColorSpace::kGray : jxl::ColorSpace::kRGB))
    throw std::runtime_error("SetSRGB failed");
  color.primaries = static_cast<jxl::Primaries>(primaries);
  color.tf.SetTransferFunction(static_cast<jxl::TransferFunction>(transfer));
  if (!color.CreateICC()) throw std::runtime_error("CreateICC failed");
  metadata->color_encoding = color;
  metadata->SetIntensityTarget(target);
  if (alpha) metadata->SetAlphaBits(32);
  jxl::ImageBundle image(metadata);
  jxl::Image3F rgb(width, height);
  jxl::ImageF a(width, height);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      uint32_t bits[4];
      for (auto& v : bits) {
        if (!(std::cin >> std::hex >> v >> std::dec))
          throw std::runtime_error("missing pixel bits");
      }
      for (size_t c = 0; c < 3; ++c) rgb.PlaneRow(c, y)[x] = Float(bits[c]);
      a.Row(y)[x] = Float(bits[3]);
    }
  }
  image.SetFromImage(std::move(rgb), color);
  if (alpha) image.SetAlpha(std::move(a), false);
  return image;
}

int main() {
  std::string mode;
  while (std::cin >> mode) {
    if (mode == "matrix") {
      int primaries;
      std::cin >> primaries;
      cmsMAT3 matrices[2];
      for (size_t i = 0; i < 2; ++i) {
        auto color = jxl::ColorEncoding::LinearSRGB(false);
        color.primaries = static_cast<jxl::Primaries>(i == 0 ? primaries : 1);
        if (!color.CreateICC()) throw std::runtime_error("CreateICC failed");
        auto profile = cmsOpenProfileFromMem(color.ICC().data(), color.ICC().size());
        const cmsTagSignature tags[3] = {cmsSigRedColorantTag, cmsSigGreenColorantTag,
                                         cmsSigBlueColorantTag};
        for (size_t c = 0; c < 3; ++c) {
          const auto* xyz = static_cast<const cmsCIEXYZ*>(cmsReadTag(profile, tags[c]));
          matrices[i].v[0].n[c] = xyz->X;
          matrices[i].v[1].n[c] = xyz->Y;
          matrices[i].v[2].n[c] = xyz->Z;
        }
        cmsCloseProfile(profile);
      }
      cmsMAT3 inv, result;
      if (!_cmsMAT3inverse(&matrices[1], &inv)) throw std::runtime_error("inverse failed");
      _cmsMAT3per(&result, &inv, &matrices[0]);
      for (const auto& row : result.v)
        std::printf("%.17g %.17g %.17g\n", row.n[0], row.n[1], row.n[2]);
      auto color = jxl::ColorEncoding::LinearSRGB(false);
      color.primaries = static_cast<jxl::Primaries>(primaries);
      const auto p = color.GetPrimaries();
      double chromaticities[9] = {p.r.x, p.g.x, p.b.x, p.r.y, p.g.y, p.b.y,
        1 - p.r.x - p.r.y, 1 - p.g.x - p.g.y, 1 - p.b.x - p.b.y};
      if (!jxl::Inv3x3Matrix(chromaticities)) throw std::runtime_error("inverse failed");
      float white[3];
      if (!jxl::CIEXYZFromWhiteCIExy(color.GetWhitePoint(), white))
        throw std::runtime_error("white conversion failed");
      const double ys[3] = {p.r.y, p.g.y, p.b.y};
      float luminances[3];
      for (size_t i = 0; i < 3; ++i)
        luminances[i] = ys[i] * (chromaticities[i*3] * white[0] +
          chromaticities[i*3+1] * white[1] + chromaticities[i*3+2] * white[2]);
      std::printf("luminances %08x %08x %08x\n", Bits(luminances[0]),
                  Bits(luminances[1]), Bits(luminances[2]));
      continue;
    }
    jxl::ImageMetadata metadata;
    float bg;
    const bool gray = mode == "frontend-gray" || mode == "score-gray";
    auto image = Read(&metadata, &bg, gray);
    if (mode == "score" || mode == "score-gray" || mode == "score-raw") {
      jxl::ImageMetadata other_metadata;
      float other_bg;
      auto other = Read(&other_metadata, &other_bg, gray);
      const auto result = ComputeSSIMULACRA2(image, other, bg);
      const double score = result.Score();
      uint64_t bits;
      std::memcpy(&bits, &score, sizeof(bits));
      std::printf("%016llx %.17g\n", static_cast<unsigned long long>(bits), score);
      if (mode == "score-raw") {
        for (const auto& scale : result.scales) {
          for (double v : scale.avg_ssim) {
            std::memcpy(&bits, &v, sizeof(bits));
            std::printf("%016llx ", static_cast<unsigned long long>(bits));
          }
          for (double v : scale.avg_edgediff) {
            std::memcpy(&bits, &v, sizeof(bits));
            std::printf("%016llx ", static_cast<unsigned long long>(bits));
          }
          std::printf("\n");
        }
      }
      continue;
    }
    if (mode != "frontend" && mode != "frontend-gray") throw std::runtime_error("unknown mode");
    // Same alpha arithmetic, in the encoded domain, as ssimulacra2.cc.
    if (image.HasAlpha()) {
      for (size_t y = 0; y < image.ysize(); ++y) {
        for (size_t x = 0; x < image.xsize(); ++x) {
          const float a = image.alpha()->Row(y)[x];
          for (size_t c = 0; c < 3; ++c) {
            float& v = image.color()->PlaneRow(c, y)[x];
            v = a * v + (1.f - a) * bg;
          }
        }
      }
    }
    image.ClearExtraChannels();
    if (!image.TransformTo(jxl::ColorEncoding::LinearSRGB(gray), jxl::GetJxlCms()))
      throw std::runtime_error("TransformTo failed");
    jxl::Image3F xyb(image.xsize(), image.ysize());
    jxl::ToXYB(image, nullptr, &xyb, jxl::GetJxlCms(), nullptr);
    for (size_t y = 0; y < image.ysize(); ++y) {
      for (size_t x = 0; x < image.xsize(); ++x) {
        const float xx = xyb.PlaneRow(0, y)[x] * 14.f + 0.42f;
        const float yy = xyb.PlaneRow(1, y)[x] + 0.01f;
        const float bb = (xyb.PlaneRow(2, y)[x] - xyb.PlaneRow(1, y)[x]) + 0.55f;
        std::printf("%08x %08x %08x %08x %08x %08x\n",
                    Bits(image.color()->PlaneRow(0, y)[x]),
                    Bits(image.color()->PlaneRow(1, y)[x]),
                    Bits(image.color()->PlaneRow(2, y)[x]), Bits(xx), Bits(yy), Bits(bb));
      }
    }
  }
}
