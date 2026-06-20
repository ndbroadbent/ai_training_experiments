//! A tiny recursive-descent arithmetic evaluator (no dependencies).
//!
//! Supports `+ - * /`, parentheses, unary minus, decimals, and ignores commas
//! (thousands separators). It powers two things: generating *correct* calculator
//! training traces, and executing `>>tool:calculate(...)` calls at inference.

use anyhow::{anyhow, bail, Result};

/// Evaluate an arithmetic expression, e.g. `"2334*9834"` -> `22952556.0`.
pub fn eval(expr: &str) -> Result<f64> {
    // Strip thousands separators so "1,000" parses as a single number.
    let expr = expr.replace(',', "");
    let tokens = lex(&expr)?;
    let mut p = Parser { tokens, pos: 0 };
    let v = p.expr()?;
    if p.pos != p.tokens.len() {
        bail!("unexpected trailing input in {expr:?}");
    }
    Ok(v)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
}

fn lex(s: &str) -> Result<Vec<Tok>> {
    let chars: Vec<char> = s.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' | '\r' | ',' => i += 1, // skip whitespace + separators
            '+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                toks.push(Tok::Star);
                i += 1;
            }
            '/' => {
                toks.push(Tok::Slash);
                i += 1;
            }
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            _ if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num: String = chars[start..i].iter().collect();
                toks.push(Tok::Num(num.parse().map_err(|_| anyhow!("bad number {num:?}"))?));
            }
            _ => bail!("unexpected character {c:?}"),
        }
    }
    Ok(toks)
}

struct Parser {
    tokens: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    // expr := term (('+' | '-') term)*
    fn expr(&mut self) -> Result<f64> {
        let mut v = self.term()?;
        while let Some(t) = self.peek() {
            match t {
                Tok::Plus => {
                    self.next();
                    v += self.term()?;
                }
                Tok::Minus => {
                    self.next();
                    v -= self.term()?;
                }
                _ => break,
            }
        }
        Ok(v)
    }

    // term := factor (('*' | '/') factor)*
    fn term(&mut self) -> Result<f64> {
        let mut v = self.factor()?;
        while let Some(t) = self.peek() {
            match t {
                Tok::Star => {
                    self.next();
                    v *= self.factor()?;
                }
                Tok::Slash => {
                    self.next();
                    let d = self.factor()?;
                    if d == 0.0 {
                        bail!("division by zero");
                    }
                    v /= d;
                }
                _ => break,
            }
        }
        Ok(v)
    }

    // factor := number | '(' expr ')' | ('+'|'-') factor
    fn factor(&mut self) -> Result<f64> {
        match self.next() {
            Some(Tok::Num(n)) => Ok(n),
            Some(Tok::Minus) => Ok(-self.factor()?),
            Some(Tok::Plus) => self.factor(),
            Some(Tok::LParen) => {
                let v = self.expr()?;
                match self.next() {
                    Some(Tok::RParen) => Ok(v),
                    _ => bail!("expected ')'"),
                }
            }
            other => bail!("unexpected token {other:?}"),
        }
    }
}

/// Format a result for display: whole numbers as integers, otherwise trimmed.
pub fn format_number(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v:.6}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        assert_eq!(eval("2334*9834").unwrap(), 22952556.0);
        assert_eq!(eval("2+3*4").unwrap(), 14.0);
        assert_eq!(eval("(2+3)*4").unwrap(), 20.0);
        assert_eq!(eval("-5+2").unwrap(), -3.0);
        assert_eq!(eval("240*15/100").unwrap(), 36.0);
        assert_eq!(eval("1,000+1").unwrap(), 1001.0);
    }

    #[test]
    fn formatting() {
        assert_eq!(format_number(22952556.0), "22952556");
        assert_eq!(format_number(3.5), "3.5");
    }

    #[test]
    fn errors() {
        assert!(eval("1/0").is_err());
        assert!(eval("2+").is_err());
        assert!(eval("(1+2").is_err());
    }
}
