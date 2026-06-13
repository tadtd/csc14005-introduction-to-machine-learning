# Slides

Slide chính của report nằm ở `slide/main.tex`.

## Cấu trúc

- `main.tex`: entry point của Beamer deck.
- `content/slides.tex`: index `\input` các section.
- `content/00_*.tex` đến `content/05_*.tex`: nội dung từng section của slide.
- `theme/`: theme Beamer local `SimpleDarkBlue`.
- `img/`: hình dùng trong slide report.
- `ref/`: BibTeX tham khảo nếu cần mở rộng citation.

## Compile

Từ root project:

```powershell
.\scripts\compile_slide.ps1
```

PDF mặc định được xuất tại `slide/build/main.pdf`.
