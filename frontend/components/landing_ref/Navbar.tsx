import React from "react"
import Link from "next/link"
import { Menu, X } from "lucide-react"

interface NavbarProps {
  scrolled: boolean
}

export const Navbar: React.FC<NavbarProps> = ({ scrolled }) => {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false)

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 border-b ${
        scrolled ? "bg-white/90 backdrop-blur-md border-slate-200 py-4" : "bg-transparent border-transparent py-6"
      }`}
    >
      <div className="container mx-auto px-6 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 cursor-pointer">
          <span className="text-xl font-bold tracking-tight text-slate-900">Proprly.</span>
        </Link>

        {/* Desktop Menu */}
        <div className="hidden md:flex items-center gap-8">
          <a href="#features" className="text-sm font-medium text-slate-600 hover:text-brand-600 transition-colors">
            Features
          </a>
          <a href="#demo" className="text-sm font-medium text-slate-600 hover:text-brand-600 transition-colors">
            How it Works
          </a>
          <a href="#pricing" className="text-sm font-medium text-slate-600 hover:text-brand-600 transition-colors">
            Pricing
          </a>
          <Link href="/login" className="text-sm font-medium text-slate-600 hover:text-brand-600 transition-colors">
            Login
          </Link>
          <Link
            href="/login"
            className="px-5 py-2.5 bg-brand-600 text-white text-sm font-semibold rounded-full hover:bg-brand-700 transition-all shadow-sm hover:shadow-md"
          >
            Get Started
          </Link>
        </div>

        {/* Mobile Menu Button */}
        <button className="md:hidden p-2 text-slate-600" onClick={() => setIsMenuOpen(!isMenuOpen)}>
          {isMenuOpen ? <X /> : <Menu />}
        </button>
      </div>

      {/* Mobile Menu Dropdown */}
      {isMenuOpen && (
        <div className="md:hidden absolute top-full left-0 w-full bg-white border-b border-slate-100 shadow-lg py-4 px-6 flex flex-col gap-4">
          <a href="#features" className="text-sm font-medium text-slate-600" onClick={() => setIsMenuOpen(false)}>
            Features
          </a>
          <a href="#demo" className="text-sm font-medium text-slate-600" onClick={() => setIsMenuOpen(false)}>
            How it Works
          </a>
          <a href="#pricing" className="text-sm font-medium text-slate-600" onClick={() => setIsMenuOpen(false)}>
            Pricing
          </a>
          <div className="h-px bg-slate-100 my-2"></div>
          <Link
            href="/login"
            className="text-sm font-medium text-slate-600"
            onClick={() => setIsMenuOpen(false)}
          >
            Login
          </Link>
          <Link
            href="/login"
            className="w-full py-3 bg-brand-600 text-white text-sm font-semibold rounded-lg text-center"
            onClick={() => setIsMenuOpen(false)}
          >
            Get Started
          </Link>
        </div>
      )}
    </nav>
  )
}


